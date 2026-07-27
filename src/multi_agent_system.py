import os
import sys
import json
from datetime import datetime

from dotenv import load_dotenv
from langfuse import observe, propagate_attributes

from src.agents.orchestrator import get_orchestrator_chain
from src.agents.hr_agent import get_hr_chain
from src.agents.tech_agent import get_tech_chain
from src.agents.finance_agent import get_finance_chain
from src.evaluator import evaluate_response
from src.langfuse_utils import (
    get_langchain_callbacks,
    get_langfuse_client,
    is_langfuse_enabled,
)

load_dotenv()


def save_to_log(data: dict, filepath: str = "results_log.json"):
    """Guarda las ejecuciones de forma acumulativa en un archivo JSON local."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        **data,
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


def ensure_string(val) -> str:
    """Garantiza la conversión de cualquier objeto a una cadena de texto (PyString)."""
    if isinstance(val, str):
        return val

    if isinstance(val, dict):
        for key in ["output", "answer", "result", "text"]:
            if key in val and isinstance(val[key], str):
                return val[key]

        return json.dumps(val, ensure_ascii=False)

    if hasattr(val, "content"):
        return str(val.content)

    return str(val)


@observe(name="route_intent", as_type="chain")
def route_intent(question: str) -> str:
    """Clasifica la consulta del usuario y retorna la etiqueta del dominio."""
    orchestrator = get_orchestrator_chain()
    callbacks = get_langchain_callbacks()

    raw_destination = orchestrator.invoke(
        {"question": question},
        config={"callbacks": callbacks} if callbacks else None,
    )

    destination = ensure_string(raw_destination).strip().lower()
    return destination if destination in ["hr", "finance", "tech"] else "out_of_scope"


@observe(name="retrieve_context", as_type="retriever")
def retrieve_context(retriever, question: str) -> tuple[str, list]:
    """Obtiene y serializa los documentos más relevantes para la pregunta."""
    callbacks = get_langchain_callbacks()
    docs = retriever.invoke(
        question,
        config={"callbacks": callbacks} if callbacks else None,
    )

    context_parts = []
    doc_sources = []

    for doc in docs:
        if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
            source = doc.metadata.get("source")
            if source:
                doc_sources.append(source)

        if hasattr(doc, "page_content"):
            context_parts.append(str(doc.page_content))
        elif isinstance(doc, dict):
            context_parts.append(doc.get("page_content", json.dumps(doc, ensure_ascii=False)))
        else:
            context_parts.append(str(doc))

    return "\n\n".join(context_parts), doc_sources


@observe(name="generate_domain_response", as_type="agent")
def generate_domain_response(agent, agent_input: dict):
    """Genera la respuesta usando el contexto recuperado por el pipeline."""
    callbacks = get_langchain_callbacks()
    config = {"callbacks": callbacks} if callbacks else None

    if config:
        return agent.invoke(agent_input, config=config)
    return agent.invoke(agent_input)


def load_test_queries(filepath: str = "test_queries.json") -> list[dict]:
    """Carga el set de consultas de prueba definido para el proyecto."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


@observe(name="routing_test_suite", as_type="chain")
def run_test_suite(filepath: str = "test_queries.json") -> dict:
    """Ejecuta pruebas de routing y resume cobertura y precisión."""
    queries = load_test_queries(filepath)
    results = []

    for item in queries:
        predicted = route_intent(item["question"])
        results.append(
            {
                "question": item["question"],
                "expected": item["expected"],
                "predicted": predicted,
                "match": predicted == item["expected"],
            }
        )

    total = len(results)
    passed = sum(1 for item in results if item["match"])
    accuracy = round((passed / total) * 100, 2) if total else 0.0
    covered_labels = sorted({item["expected"] for item in results})

    return {
        "total_queries": total,
        "passed": passed,
        "failed": total - passed,
        "routing_accuracy_percent": accuracy,
        "covered_labels": covered_labels,
        "results": results,
    }


@observe(name="multi_agent_pipeline", as_type="chain")
def run_pipeline(question: str, session_id: str = "default_session"):

    clean_question = ensure_string(question)
    langfuse_client = get_langfuse_client() if is_langfuse_enabled() else None

    with propagate_attributes(
        session_id=session_id,
        trace_name="multi_agent_pipeline",
        metadata={"entrypoint": "run_pipeline"},
        tags=["multi-agent", "langchain", "rag"],
    ):
        if langfuse_client is not None:
            langfuse_client.update_current_span(
                input={"question": clean_question, "session_id": session_id}
            )

        # 1. Router
        destination = route_intent(clean_question)

        # 2. Fuera de alcance
        if destination == "out_of_scope":

            response_text = (
                "Lo siento, esa consulta está fuera del ámbito corporativo "
                "de Recursos Humanos, Finanzas o Soporte Técnico."
            )

            result = {
                "question": clean_question,
                "destination": destination,
                "response": response_text,
                "evaluation": {
                    "score_general": 0,
                    "dimensiones": {
                        "relevancia": 0,
                        "completitud": 0,
                        "fidelidad": 0,
                    },
                    "justificacion": "Consulta fuera del dominio soportado por los agentes.",
                },
            }

            if langfuse_client is not None:
                langfuse_client.update_current_span(
                    metadata={"destination": destination, "out_of_scope": True}
                )
                langfuse_client.score_current_trace(
                    name="score_general",
                    value=0,
                    data_type="NUMERIC",
                    comment="Consulta clasificada como fuera de alcance.",
                    metadata={"destination": destination},
                )
                for metric_name in ("relevancia", "completitud", "fidelidad"):
                    langfuse_client.score_current_trace(
                        name=metric_name,
                        value=0,
                        data_type="NUMERIC",
                        comment="Consulta clasificada como fuera de alcance.",
                        metadata={"destination": destination},
                    )
                langfuse_client.update_current_span(output=result)
                langfuse_client.flush()

            save_to_log(result)

            return result

        # 3. Selección del agente
        if destination == "hr":
            agent_res = get_hr_chain()

        elif destination == "tech":
            agent_res = get_tech_chain()

        else:
            agent_res = get_finance_chain()

        # 4. Recuperación única del contexto
        agent, retriever = agent_res
        context_text, doc_sources = retrieve_context(retriever, clean_question)
        agent_input = {
            "question": clean_question,
            "context": context_text,
        }

        if langfuse_client is not None:
            langfuse_client.update_current_span(
                metadata={
                    "destination": destination,
                    "retrieved_sources": doc_sources,
                    "retrieved_context_characters": len(context_text),
                    "has_retriever": True,
                }
            )

        # 5. Ejecutar agente
        response = generate_domain_response(agent, agent_input)
        response_text = ensure_string(response)

        # 6. Evaluación
        eval_res = evaluate_response(
            question=clean_question,
            response=response_text,
            context=context_text,
            destination=destination,
        )

        result = {
            "question": clean_question,
            "destination": destination,
            "response": response_text,
            "evaluation": eval_res,
        }

        if langfuse_client is not None:
            langfuse_client.update_current_span(output=result)
            langfuse_client.flush()

        save_to_log(result)

        return result


if __name__ == "__main__":

    if "--run-tests" in sys.argv:
        summary = run_test_suite()
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        raise SystemExit(0)

    print("\n🤖 SISTEMA MULTIAGENTE INTERACTIVO ACTIVO")
    print("Escribí tu pregunta y presioná Enter.")
    print("Para salir escribí 'salir', 'exit' o 'q'.")

    while True:

        try:

            user_input = input("\n📝 Ingresá tu pregunta: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["salir", "exit", "q"]:
                print("¡Hasta luego!")
                break

            result = run_pipeline(user_input)

            eval_data = result.get("evaluation", {})

            print("\n" + "=" * 50)
            print(f"🎯 Destino: {result['destination'].upper()}")
            print(f"💬 Respuesta: {result['response']}")
            print("-" * 50)

            if isinstance(eval_data, dict):

                score = eval_data.get('score_general') or eval_data.get('score')
                print(f"⭐ Score: {score if score is not None else 'N/A'}")

                just = eval_data.get("justificacion") or eval_data.get("reasoning") or eval_data.get("feedback")

                if just:
                    print(f"📝 Justificación: {just}")

            else:
                print(eval_data)

            print("=" * 50)

        except KeyboardInterrupt:
            print("\nEjecución cancelada.")
            break

        except Exception as e:
            print(f"\n❌ Error: {e}")
