from pydantic import BaseModel, ConfigDict, Field

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langfuse import observe

from src.langfuse_utils import (
    get_langchain_callbacks,
    get_langfuse_client,
    is_langfuse_enabled,
)


class EvaluationDimensions(BaseModel):
    """Puntajes validados para las dimensiones de calidad."""

    model_config = ConfigDict(extra="forbid")

    relevancia: int = Field(
        ge=1,
        le=10,
        description="Qué tan directamente responde la consulta original.",
    )
    completitud: int = Field(
        ge=1,
        le=10,
        description="Qué tan completa es la respuesta según el contexto disponible.",
    )
    fidelidad: int = Field(
        ge=1,
        le=10,
        description="Qué tan respaldadas están las afirmaciones por el contexto.",
    )


class EvaluationJudgment(BaseModel):
    """Salida estructurada que debe producir el LLM evaluador."""

    model_config = ConfigDict(extra="forbid")

    dimensiones: EvaluationDimensions
    justificacion: str = Field(min_length=10, max_length=1000)


def build_evaluation_result(judgment: EvaluationJudgment) -> dict:
    """Convierte el juicio validado al contrato público del evaluator."""
    dimensions = judgment.dimensiones.model_dump()
    score_general = round(sum(dimensions.values()) / len(dimensions), 2)

    return {
        "status": "evaluated",
        "score_general": score_general,
        "dimensiones": dimensions,
        "justificacion": judgment.justificacion,
    }


def build_evaluation_error(error: Exception) -> dict:
    """Representa un fallo técnico sin inventar puntajes de calidad."""
    return {
        "status": "evaluation_error",
        "score_general": None,
        "dimensiones": {
            "relevancia": None,
            "completitud": None,
            "fidelidad": None,
        },
        "justificacion": "La respuesta no pudo ser evaluada automáticamente.",
        "error_type": type(error).__name__,
        "error_message": str(error),
    }


def build_not_applicable_evaluation(reason: str) -> dict:
    """Representa una salida que no corresponde evaluar como respuesta RAG."""
    return {
        "status": "not_applicable",
        "score_general": None,
        "dimensiones": {
            "relevancia": None,
            "completitud": None,
            "fidelidad": None,
        },
        "justificacion": reason,
    }


def record_evaluation_scores(
    result: dict,
    question: str,
    destination: str,
    context: str,
) -> None:
    """Registra en Langfuse scores válidos o el estado técnico del evaluator."""
    if not is_langfuse_enabled():
        return

    client = get_langfuse_client()
    metadata = {
        "question": str(question),
        "destination": destination,
        "context_characters": len(str(context)),
        "evaluation_status": result["status"],
    }

    if result["status"] == "evaluated":
        client.score_current_trace(
            name="evaluation_succeeded",
            value=1,
            data_type="BOOLEAN",
            metadata=metadata,
        )
        client.score_current_trace(
            name="score_general",
            value=float(result["score_general"]),
            data_type="NUMERIC",
            comment=result["justificacion"],
            metadata=metadata,
        )

        for metric_name, value in result["dimensiones"].items():
            client.score_current_trace(
                name=metric_name,
                value=float(value),
                data_type="NUMERIC",
                comment=result["justificacion"],
                metadata=metadata,
            )

        client.score_current_trace(
            name="justificacion_evaluador",
            value=result["justificacion"],
            data_type="TEXT",
            metadata=metadata,
        )
        return

    client.score_current_trace(
        name="evaluation_succeeded",
        value=0,
        data_type="BOOLEAN",
        comment=result["justificacion"],
        metadata=metadata,
    )

    if result.get("error_message"):
        client.score_current_trace(
            name="evaluation_error",
            value=result["error_message"],
            data_type="TEXT",
            metadata=metadata,
        )


@observe(name="response_evaluator", as_type="evaluator")
def evaluate_response(
    question: str,
    response: str,
    context: str = "",
    destination: str = "",
) -> dict:
    """Evalúa una respuesta RAG con schema estricto y rúbrica calibrada."""
    try:
        parser = PydanticOutputParser(pydantic_object=EvaluationJudgment)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Eres un auditor de calidad para respuestas RAG corporativas.

Evalúa exclusivamente la RESPUESTA usando la PREGUNTA y el CONTEXTO recuperado.
No completes información faltante ni asumas políticas externas.

RÚBRICA (enteros de 1 a 10):

RELEVANCIA
- 10: responde directamente la intención principal.
- 5: responde solo una parte o incluye contenido mayormente tangencial.
- 1: no responde la pregunta.

COMPLETITUD
- 10: incluye toda la información necesaria que está disponible en el contexto.
- 5: omite información importante o no explica una limitación relevante.
- 1: carece de casi toda la información necesaria.

FIDELIDAD
- 10: todas las afirmaciones verificables están respaldadas por el contexto.
- 5: contiene afirmaciones ambiguas o parcialmente respaldadas.
- 1: contradice el contexto o inventa información importante.

Si el contexto es insuficiente y la respuesta lo reconoce correctamente, no
penalices la fidelidad por abstenerse. Sí evalúa si explica adecuadamente la
limitación. No calcules un score general: el sistema lo calcula en código.

{format_instructions}""",
                ),
                (
                    "human",
                    """DESTINO CLASIFICADO:
{destination}

PREGUNTA:
<question>
{question}
</question>

RESPUESTA:
<response>
{response}
</response>

CONTEXTO RECUPERADO:
<context>
{context}
</context>""",
                ),
            ]
        ).partial(format_instructions=parser.get_format_instructions())

        chain = prompt | llm | parser
        callbacks = get_langchain_callbacks()
        judgment = chain.invoke(
            {
                "question": str(question),
                "response": str(response),
                "context": str(context),
                "destination": str(destination),
            },
            config={"callbacks": callbacks} if callbacks else None,
        )
        result = build_evaluation_result(judgment)
    except Exception as error:
        result = build_evaluation_error(error)

    try:
        record_evaluation_scores(
            result=result,
            question=str(question),
            destination=str(destination),
            context=str(context),
        )
    except Exception as telemetry_error:
        result["telemetry_error"] = {
            "error_type": type(telemetry_error).__name__,
            "error_message": str(telemetry_error),
        }

    return result
