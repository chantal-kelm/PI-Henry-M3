from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langfuse import observe

from src.langfuse_utils import get_langchain_callbacks, get_langfuse_client, is_langfuse_enabled


@observe(name="response_evaluator", as_type="evaluator")
def evaluate_response(
    question: str,
    response: str,
    context: str = "",
    destination: str = "",
) -> dict:
    """
    Audita la respuesta generada por el agente utilizando un LLM evaluador.
    Evalúa relevancia, completitud y fidelidad.
    """

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """Eres un auditor de calidad experto en evaluar sistemas RAG y Multiagente.

Evalúa la respuesta dada al usuario según las siguientes 3 dimensiones (escala de 1 a 10):

1. relevancia
2. completitud
3. fidelidad

Calcula el score_general como el promedio de las tres dimensiones.

Responde ÚNICAMENTE con este JSON:

{{
  "score_general": int,
  "dimensiones": {{
    "relevancia": int,
    "completitud": int,
    "fidelidad": int
  }},
  "justificacion": "Explicación breve"
}}
""",
        ),
        (
            "human",
            """Pregunta: {question}

Respuesta:
{response}

Contexto:
{context}""",
        ),
    ])

    chain = prompt | llm | JsonOutputParser()
    callbacks = get_langchain_callbacks()

    try:
        result = chain.invoke(
            {
                "question": str(question),
                "response": str(response),
                "context": str(context),
            },
            config={"callbacks": callbacks} if callbacks else None,
        )

        if is_langfuse_enabled():
            client = get_langfuse_client()
            dimensions = result.get("dimensiones", {})
            metadata = {
                "question": str(question),
                "destination": destination,
                "context_characters": len(str(context)),
            }

            client.score_current_trace(
                name="score_general",
                value=float(result.get("score_general", 0)),
                data_type="NUMERIC",
                comment=result.get("justificacion"),
                metadata=metadata,
            )

            for metric_name in ("relevancia", "completitud", "fidelidad"):
                if metric_name in dimensions:
                    client.score_current_trace(
                        name=metric_name,
                        value=float(dimensions[metric_name]),
                        data_type="NUMERIC",
                        comment=result.get("justificacion"),
                        metadata=metadata,
                    )

            if result.get("justificacion"):
                client.score_current_trace(
                    name="justificacion_evaluador",
                    value=result["justificacion"],
                    data_type="TEXT",
                    metadata=metadata,
                )

        return result

    except Exception as e:
        fallback_result = {
            "score_general": 5,
            "dimensiones": {
                "relevancia": 5,
                "completitud": 5,
                "fidelidad": 5,
            },
            "justificacion": f"Error en la evaluación automática: {e}",
        }

        if is_langfuse_enabled():
            client = get_langfuse_client()
            metadata = {
                "question": str(question),
                "destination": destination,
                "context_characters": len(str(context)),
                "fallback": True,
            }

            client.score_current_trace(
                name="score_general",
                value=5.0,
                data_type="NUMERIC",
                comment=fallback_result["justificacion"],
                metadata=metadata,
            )

            for metric_name in ("relevancia", "completitud", "fidelidad"):
                client.score_current_trace(
                    name=metric_name,
                    value=5.0,
                    data_type="NUMERIC",
                    comment=fallback_result["justificacion"],
                    metadata=metadata,
                )

        return fallback_result
