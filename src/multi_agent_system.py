import os
import sys
import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langfuse import observe, propagate_attributes
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)

from src.agents.orchestrator import get_orchestrator_chain
from src.agents.hr_agent import get_hr_rag_chain
from src.agents.tech_agent import get_tech_rag_chain
from src.agents.finance_agent import get_finance_rag_chain
from src.evaluator import build_not_applicable_evaluation, evaluate_response
from src.langfuse_utils import (
    get_langchain_callbacks,
    get_langfuse_client,
    is_langfuse_enabled,
)

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST_QUERIES_PATH = PROJECT_ROOT / "test_queries.json"
SUPPORTED_DESTINATIONS = frozenset({"hr", "finance", "tech", "out_of_scope"})
REQUIRED_TEST_DESTINATIONS = SUPPORTED_DESTINATIONS
MIN_TEST_QUERIES = 10
DEFAULT_MIN_ROUTING_ACCURACY = 90.0


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


def normalize_destination(raw_destination) -> str:
    """Normaliza la salida del clasificador al contrato público del router."""
    destination = ensure_string(raw_destination).strip().lower()
    return destination if destination in SUPPORTED_DESTINATIONS else "out_of_scope"


@observe(name="route_intent", as_type="chain")
def route_intent(question: str) -> str:
    """Clasifica la consulta del usuario y retorna la etiqueta del dominio."""
    orchestrator = get_orchestrator_chain()
    callbacks = get_langchain_callbacks()

    raw_destination = orchestrator.invoke(
        {"question": question},
        config={"callbacks": callbacks} if callbacks else None,
    )

    return normalize_destination(raw_destination)


def build_out_of_scope_response(route_input: dict) -> dict:
    """Construye la salida estática de la rama fuera de alcance."""
    return {
        "question": route_input["question"],
        "destination": "out_of_scope",
        "context": "",
        "doc_sources": [],
        "response": (
            "Lo siento, esa consulta está fuera del ámbito corporativo "
            "de Recursos Humanos, Finanzas o Soporte Técnico."
        ),
    }


def get_conditional_agent_router() -> RunnableBranch:
    """Enruta y ejecuta el agente RAG LCEL completo del dominio seleccionado."""
    return RunnableBranch(
        (
            lambda route_input: route_input["destination"] == "hr",
            RunnableLambda(lambda _route_input: get_hr_rag_chain()).with_config(
                run_name="execute_hr_rag_agent"
            ),
        ),
        (
            lambda route_input: route_input["destination"] == "tech",
            RunnableLambda(lambda _route_input: get_tech_rag_chain()).with_config(
                run_name="execute_tech_rag_agent"
            ),
        ),
        (
            lambda route_input: route_input["destination"] == "finance",
            RunnableLambda(lambda _route_input: get_finance_rag_chain()).with_config(
                run_name="execute_finance_rag_agent"
            ),
        ),
        RunnableLambda(build_out_of_scope_response).with_config(
            run_name="handle_out_of_scope"
        ),
    )


def get_multi_agent_orchestrator() -> Runnable:
    """
    Construye el workflow dinámico completo de clasificación y delegación.

    La salida del clasificador alimenta un RunnableBranch que ejecuta dentro de
    la rama elegida la recuperación y generación del agente RAG especializado.
    """
    intent_classifier = (
        get_orchestrator_chain()
        | RunnableLambda(normalize_destination).with_config(
            run_name="normalize_destination"
        )
    ).with_config(run_name="route_intent")

    return (
        RunnablePassthrough.assign(destination=intent_classifier)
        | get_conditional_agent_router()
    ).with_config(run_name="multi_agent_orchestrator")


def validate_test_queries(queries: object) -> list[dict]:
    """Valida el contrato mínimo del dataset de aceptación de routing."""
    if not isinstance(queries, list):
        raise ValueError("El dataset de routing debe ser una lista JSON.")

    if len(queries) < MIN_TEST_QUERIES:
        raise ValueError(
            f"El dataset debe contener al menos {MIN_TEST_QUERIES} consultas."
        )

    validated_queries = []
    seen_questions = set()

    for index, item in enumerate(queries):
        if not isinstance(item, dict):
            raise ValueError(f"La consulta en posición {index} debe ser un objeto.")

        question = item.get("question")
        expected = item.get("expected")

        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                f"La consulta en posición {index} debe tener una pregunta no vacía."
            )

        if expected not in SUPPORTED_DESTINATIONS:
            raise ValueError(
                f"La consulta en posición {index} tiene una categoría inválida: "
                f"{expected!r}."
            )

        normalized_question = question.strip()
        if normalized_question in seen_questions:
            raise ValueError(f"Pregunta duplicada en el dataset: {normalized_question}")

        seen_questions.add(normalized_question)
        validated_queries.append(
            {"question": normalized_question, "expected": expected}
        )

    covered_labels = {item["expected"] for item in validated_queries}
    missing_labels = REQUIRED_TEST_DESTINATIONS - covered_labels

    if missing_labels:
        missing = ", ".join(sorted(missing_labels))
        raise ValueError(f"El dataset no cubre las categorías requeridas: {missing}.")

    return validated_queries


def load_test_queries(
    filepath: str | Path = DEFAULT_TEST_QUERIES_PATH,
) -> list[dict]:
    """Carga y valida el set de consultas de aceptación del proyecto."""
    path = Path(filepath)
    with path.open("r", encoding="utf-8") as file:
        return validate_test_queries(json.load(file))


def evaluate_routing_queries(
    queries: list[dict],
    predictor: Callable[[str], str],
    min_accuracy_percent: float = DEFAULT_MIN_ROUTING_ACCURACY,
) -> dict:
    """Evalúa un predictor de routing con un criterio explícito de aprobación."""
    if not 0 <= min_accuracy_percent <= 100:
        raise ValueError("El umbral de precisión debe estar entre 0 y 100.")

    results = []

    for item in queries:
        predicted = predictor(item["question"])
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
    meets_threshold = accuracy >= min_accuracy_percent

    return {
        "status": "passed" if meets_threshold else "failed",
        "meets_threshold": meets_threshold,
        "minimum_accuracy_percent": min_accuracy_percent,
        "total_queries": total,
        "passed": passed,
        "failed": total - passed,
        "routing_accuracy_percent": accuracy,
        "covered_labels": covered_labels,
        "results": results,
    }


def routing_test_exit_code(summary: dict) -> int:
    """Retorna un código de proceso seguro según el criterio de aprobación."""
    return 0 if summary.get("meets_threshold") is True else 1


@observe(name="routing_test_suite", as_type="chain")
def run_test_suite(
    filepath: str | Path = DEFAULT_TEST_QUERIES_PATH,
    min_accuracy_percent: float = DEFAULT_MIN_ROUTING_ACCURACY,
) -> dict:
    """Ejecuta la prueba de aceptación live del router."""
    queries = load_test_queries(filepath)
    return evaluate_routing_queries(
        queries=queries,
        predictor=route_intent,
        min_accuracy_percent=min_accuracy_percent,
    )


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

        # 1. Workflow LangChain completo:
        # clasificación -> routing condicional -> recuperación -> generación.
        callbacks = get_langchain_callbacks()
        orchestrator = get_multi_agent_orchestrator()
        agent_result = orchestrator.invoke(
            {"question": clean_question},
            config={
                "callbacks": callbacks,
                "metadata": {"session_id": session_id},
            },
        )
        destination = agent_result["destination"]
        response_text = ensure_string(agent_result["response"])

        # 2. Fuera de alcance
        if destination == "out_of_scope":
            result = {
                "question": clean_question,
                "destination": destination,
                "response": response_text,
                "evaluation": build_not_applicable_evaluation(
                    "Consulta fuera del dominio soportado por los agentes."
                ),
            }

            if langfuse_client is not None:
                langfuse_client.update_current_span(
                    metadata={"destination": destination, "out_of_scope": True}
                )
                langfuse_client.score_current_trace(
                    name="evaluation_applicable",
                    value=0,
                    data_type="BOOLEAN",
                    comment="Las consultas fuera de alcance no generan respuesta RAG.",
                    metadata={"destination": destination},
                )
                langfuse_client.score_current_trace(
                    name="out_of_scope_handled",
                    value=1,
                    data_type="BOOLEAN",
                    comment="Consulta rechazada correctamente por el router.",
                    metadata={"destination": destination},
                )
                langfuse_client.update_current_span(output=result)
                langfuse_client.flush()

            save_to_log(result)

            return result

        # 3. La rama RAG ya ejecutó una única recuperación y la generación.
        context_text = ensure_string(agent_result["context"])
        doc_sources = agent_result["doc_sources"]

        if langfuse_client is not None:
            langfuse_client.update_current_span(
                metadata={
                    "destination": destination,
                    "retrieved_sources": doc_sources,
                    "retrieved_context_characters": len(context_text),
                    "has_retriever": True,
                }
            )

        # 4. Evaluación
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
        raise SystemExit(routing_test_exit_code(summary))

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
