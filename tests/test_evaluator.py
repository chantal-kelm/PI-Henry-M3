import unittest
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pydantic import ValidationError

from src.evaluator import (
    EvaluationDimensions,
    EvaluationJudgment,
    build_evaluation_error,
    build_evaluation_result,
    build_not_applicable_evaluation,
    evaluate_response,
    record_evaluation_scores,
)


class FakeLangfuseClient:
    def __init__(self):
        self.calls = []

    def score_current_trace(self, **kwargs):
        self.calls.append(kwargs)


class EvaluationSchemaTests(unittest.TestCase):
    def test_builds_general_score_from_validated_dimensions(self):
        judgment = EvaluationJudgment(
            dimensiones=EvaluationDimensions(
                relevancia=9,
                completitud=7,
                fidelidad=9,
            ),
            justificacion="La respuesta es relevante y está respaldada.",
        )

        result = build_evaluation_result(judgment)

        self.assertEqual(result["status"], "evaluated")
        self.assertEqual(result["score_general"], 8.33)
        self.assertEqual(
            result["dimensiones"],
            {"relevancia": 9, "completitud": 7, "fidelidad": 9},
        )

    def test_rejects_scores_outside_one_to_ten(self):
        with self.assertRaises(ValidationError):
            EvaluationDimensions(
                relevancia=11,
                completitud=7,
                fidelidad=9,
            )

    def test_rejects_extra_dimensions(self):
        with self.assertRaises(ValidationError):
            EvaluationDimensions(
                relevancia=9,
                completitud=7,
                fidelidad=9,
                estilo=10,
            )


class EvaluationStatusTests(unittest.TestCase):
    def test_technical_error_has_no_quality_score(self):
        result = build_evaluation_error(RuntimeError("provider unavailable"))

        self.assertEqual(result["status"], "evaluation_error")
        self.assertIsNone(result["score_general"])
        self.assertTrue(
            all(value is None for value in result["dimensiones"].values())
        )
        self.assertEqual(result["error_type"], "RuntimeError")

    def test_not_applicable_has_no_quality_score(self):
        result = build_not_applicable_evaluation("Fuera de alcance.")

        self.assertEqual(result["status"], "not_applicable")
        self.assertIsNone(result["score_general"])
        self.assertNotIn("error_type", result)

    @patch("src.evaluator.is_langfuse_enabled", return_value=False)
    @patch("src.evaluator.get_langchain_callbacks", return_value=[])
    @patch("src.evaluator.ChatOpenAI")
    def test_evaluate_response_parses_valid_structured_output(
        self,
        chat_openai,
        _callbacks,
        _langfuse_enabled,
    ):
        chat_openai.return_value = FakeListChatModel(
            responses=[
                """{
                    "dimensiones": {
                        "relevancia": 10,
                        "completitud": 8,
                        "fidelidad": 9
                    },
                    "justificacion": "La respuesta aborda la consulta y usa el contexto."
                }"""
            ]
        )

        result = evaluate_response.__wrapped__(
            question="¿Cuál es la política?",
            response="La política indica cinco días.",
            context="La política indica cinco días.",
            destination="hr",
        )

        self.assertEqual(result["status"], "evaluated")
        self.assertEqual(result["score_general"], 9.0)

    @patch("src.evaluator.is_langfuse_enabled", return_value=False)
    @patch("src.evaluator.get_langchain_callbacks", return_value=[])
    @patch("src.evaluator.ChatOpenAI")
    def test_invalid_llm_score_becomes_evaluation_error(
        self,
        chat_openai,
        _callbacks,
        _langfuse_enabled,
    ):
        chat_openai.return_value = FakeListChatModel(
            responses=[
                """{
                    "dimensiones": {
                        "relevancia": 42,
                        "completitud": 8,
                        "fidelidad": 9
                    },
                    "justificacion": "Este resultado contiene un score inválido."
                }"""
            ]
        )

        result = evaluate_response.__wrapped__(
            question="¿Cuál es la política?",
            response="Respuesta",
            context="Contexto",
            destination="hr",
        )

        self.assertEqual(result["status"], "evaluation_error")
        self.assertIsNone(result["score_general"])


class EvaluationScoringTests(unittest.TestCase):
    @patch("src.evaluator.is_langfuse_enabled", return_value=True)
    @patch("src.evaluator.get_langfuse_client")
    def test_valid_evaluation_records_numeric_scores(
        self,
        get_client,
        _langfuse_enabled,
    ):
        client = FakeLangfuseClient()
        get_client.return_value = client
        result = {
            "status": "evaluated",
            "score_general": 8.33,
            "dimensiones": {
                "relevancia": 9,
                "completitud": 7,
                "fidelidad": 9,
            },
            "justificacion": "La respuesta está mayormente respaldada.",
        }

        record_evaluation_scores(result, "Pregunta", "hr", "Contexto")

        calls_by_name = {call["name"]: call for call in client.calls}
        self.assertEqual(calls_by_name["evaluation_succeeded"]["value"], 1)
        self.assertEqual(calls_by_name["score_general"]["value"], 8.33)
        self.assertEqual(calls_by_name["relevancia"]["value"], 9.0)
        self.assertEqual(calls_by_name["completitud"]["value"], 7.0)
        self.assertEqual(calls_by_name["fidelidad"]["value"], 9.0)
        self.assertIn("justificacion_evaluador", calls_by_name)

    @patch("src.evaluator.is_langfuse_enabled", return_value=True)
    @patch("src.evaluator.get_langfuse_client")
    def test_evaluation_error_does_not_record_numeric_quality_scores(
        self,
        get_client,
        _langfuse_enabled,
    ):
        client = FakeLangfuseClient()
        get_client.return_value = client
        result = build_evaluation_error(RuntimeError("provider unavailable"))

        record_evaluation_scores(result, "Pregunta", "hr", "Contexto")

        calls_by_name = {call["name"]: call for call in client.calls}
        self.assertEqual(calls_by_name["evaluation_succeeded"]["value"], 0)
        self.assertIn("evaluation_error", calls_by_name)
        self.assertNotIn("score_general", calls_by_name)
        self.assertNotIn("relevancia", calls_by_name)
        self.assertNotIn("completitud", calls_by_name)
        self.assertNotIn("fidelidad", calls_by_name)


if __name__ == "__main__":
    unittest.main()
