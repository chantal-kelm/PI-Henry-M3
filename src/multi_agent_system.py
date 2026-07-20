import os
import json
from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from src.agents.orchestrator import get_orchestrator_chain
from src.agents.hr_agent import get_hr_chain
from src.agents.tech_agent import get_tech_chain
from src.agents.finance_agent import get_finance_chain
from src.evaluator import evaluate_response
from datetime import datetime

load_dotenv()

# 2. Inicializamos el cliente global con debug=True
langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
    debug=False
)

def save_to_log(data: dict, filepath: str = "results_log.json"):
    """Guarda las ejecuciones de forma acumulativa en un archivo JSON."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        **data
    }
    logs = []
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except Exception:
            logs = []
            
    logs.append(log_entry)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

def run_pipeline(question: str):
    # 1. Inicializamos el cliente principal de Langfuse
    langfuse_client = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST")
    )
    
    # 2. Creamos la traza padre
    trace = langfuse_client.trace(
        name="multi_agent_pipeline",
        input=question
    )
    
    # 3. Handler nativo para LangChain derivado de esta traza
    langfuse_handler = trace.get_langchain_handler()

    orchestrator = get_orchestrator_chain()
    
    try:
        q_clean = question.lower()
        
        hr_keywords = ["vacaciones", "ley", "rrhh", "recursos humanos", "dias", "días", "licencia", "sueldo"]
        finance_keywords = ["viaticos", "viáticos", "gastos", "finanzas", "reembolso", "limite", "límite", "factura", "pago", "saldo"]
        tech_keywords = ["vpn", "computadora", "error", "credenciales", "enciende", "password", "sistema", "soporte"]
        
        if any(w in q_clean for w in hr_keywords):
            destination = "hr"
        elif any(w in q_clean for w in finance_keywords):
            destination = "finance"
        elif any(w in q_clean for w in tech_keywords):
            destination = "tech"
        else:
            print("--- ROUTER: Pregunta fuera de ámbito (Out of Scope) ---")
            out_res = {
                "question": question,
                "destination": "out_of_scope",
                "response": "Lo siento, pero esa consulta está fuera del ámbito de este sistema. Solo puedo ayudarte con temas de Tech, RH o Finance.",
                "evaluation": {"score_general": 0}
            }
            trace.update(output=out_res["response"])
            save_to_log(out_res)
            return out_res
            
        print(f"--- ROUTER (Bypass Activo): {destination.upper()} ---")
    except Exception as e:
        print(f"Error en el router: {e}")
        destination = "tech"
        
    if destination == "hr":
        agent = get_hr_chain()
    elif destination == "tech":
        agent = get_tech_chain()
    else:
        agent = get_finance_chain()
        
    try:
        # 4. Invocar el agente pasando el handler
        response = agent.invoke(question, config={"callbacks": [langfuse_handler]})
        
        # Actualizamos la traza con la respuesta final
        trace.update(output=response)
        
        # 5. Evaluamos pasando el objeto trace
        eval_res = evaluate_response(
            question=question, 
            response=response, 
            context="", 
            trace=trace
        )
        
        result_data = {
            "question": question,
            "destination": destination,
            "response": response,
            "evaluation": eval_res
        }
        
        save_to_log(result_data)
        return result_data

    finally:
        # ESTO GARANTIZA QUE SÍ O SÍ SE TRANSMITAN LOS DATOS A LANGFUSE
        langfuse_client.flush()

if __name__ == "__main__":
    print("=" * 50)
    print("🤖 SISTEMA MULTIAGENTE INTERACTIVO ACTIVO")
    print("Escribí tu pregunta y presioná Enter.")
    print("Para salir, escribí 'salir', 'exit' o 'q'.")
    print("=" * 50)
    
    while True:
        try:
            user_question = input("\n📝 Ingresá tu pregunta: ").strip()
            
            if user_question.lower() in ["salir", "exit", "q"]:
                print("\n👋 ¡Nos vemos! Cerrando el sistema...")
                break
                
            if not user_question:
                continue
                
            print("\nProcesando...")
            res = run_pipeline(user_question)
            
            print("-" * 40)
            print(f"🎯 Destino Detectado: {res['destination'].upper()}")
            print(f"🤖 Respuesta del Agente:\n{res['response']}")
            
            score = res['evaluation'].get('score_general', res['evaluation'].get('score', 0))
            print(f"📊 Evaluación (Score): {score}")
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\n\n👋 Ejecución cancelada por el usuario. ¡Adiós!")
            break
        except Exception as e:
            print(f"❌ Ocurrió un error al procesar la pregunta: {e}")