import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

def evaluate_response(question: str, response: str, context: str = "", trace=None) -> dict:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Eres un auditor de calidad experto en evaluar sistemas RAG y Multiagente.
Evalúa la respuesta dada al usuario según las siguientes 3 dimensiones (escala de 1 a 10 cada una):

1. **relevancia**: ¿La respuesta se enfoca directamente en lo que preguntó el usuario?
2. **completitud**: ¿La respuesta responde de forma exhaustiva con la información requerida?
3. **fidelidad**: ¿La respuesta se mantiene fiel al contexto o políticas sin inventar/alucinar datos?

Calcula el `score_general` como el promedio ponderado de las 3 dimensiones.

Debes responder ÚNICAMENTE en formato JSON válido con esta estructura exacta:
{{
  "score_general": int,
  "dimensiones": {{
    "relevancia": int,
    "completitud": int,
    "fidelidad": int
  }},
  "justificacion": "Explicación breve del puntaje asignado"
}}"""),
        ("human", "Pregunta: {question}\nRespuesta del Agente: {response}\nContexto: {context}")
    ])
    
    chain = prompt | llm | JsonOutputParser()
    
    try:
        res = chain.invoke({"question": question, "response": response, "context": context})
        
        # --- REGISTRO DIRECTO DE SCORES SOBRE LA TRAZA ---
        if trace:
            dims = res.get("dimensiones", {})
            comment = res.get("justificacion", "")
            
            trace.score(
                name="relevance",
                value=dims.get("relevancia", 0) / 10.0,
                comment=comment
            )
            trace.score(
                name="completeness",
                value=dims.get("completitud", 0) / 10.0,
                comment=comment
            )
            trace.score(
                name="accuracy",
                value=dims.get("fidelidad", 0) / 10.0,
                comment=comment
            )
            trace.score(
                name="overall_quality",
                value=res.get("score_general", 0) / 10.0,
                comment=comment
            )
            
        return res

    except Exception as e:
        error_res = {
            "score_general": 5,
            "dimensiones": {"relevancia": 5, "completitud": 5, "fidelidad": 5},
            "justificacion": f"Error en la evaluación automática: {e}"
        }
        if trace:
            trace.score(name="overall_quality", value=0.5, comment=str(e))
            
        return error_res