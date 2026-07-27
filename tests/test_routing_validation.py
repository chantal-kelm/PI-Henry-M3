import unittest

from src.multi_agent_system import (
    DEFAULT_MIN_ROUTING_ACCURACY,
    REQUIRED_TEST_DESTINATIONS,
    evaluate_routing_queries,
    load_test_queries,
    routing_test_exit_code,
    validate_test_queries,
)


class RoutingDatasetTests(unittest.TestCase):
    def test_project_dataset_meets_contract(self):
        queries = load_test_queries()

        self.assertGreaterEqual(len(queries), 10)
        self.assertEqual(
            {item["expected"] for item in queries},
            REQUIRED_TEST_DESTINATIONS,
        )
        self.assertEqual(
            len({item["question"] for item in queries}),
            len(queries),
        )

    def test_rejects_dataset_with_missing_categories(self):
        queries = [
            {"question": f"Consulta de HR {index}", "expected": "hr"}
            for index in range(10)
        ]

        with self.assertRaisesRegex(ValueError, "no cubre las categorías"):
            validate_test_queries(queries)

    def test_rejects_invalid_destination(self):
        queries = load_test_queries()
        queries[0] = {**queries[0], "expected": "legal"}

        with self.assertRaisesRegex(ValueError, "categoría inválida"):
            validate_test_queries(queries)

    def test_rejects_duplicate_questions(self):
        queries = load_test_queries()
        queries[-1] = {**queries[-1], "question": queries[0]["question"]}

        with self.assertRaisesRegex(ValueError, "Pregunta duplicada"):
            validate_test_queries(queries)


class RoutingEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.queries = load_test_queries()
        self.expected_by_question = {
            item["question"]: item["expected"] for item in self.queries
        }

    def test_perfect_predictor_passes(self):
        summary = evaluate_routing_queries(
            queries=self.queries,
            predictor=self.expected_by_question.__getitem__,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertTrue(summary["meets_threshold"])
        self.assertEqual(summary["routing_accuracy_percent"], 100.0)
        self.assertEqual(summary["failed"], 0)

    def test_predictor_below_threshold_fails(self):
        summary = evaluate_routing_queries(
            queries=self.queries,
            predictor=lambda _question: "out_of_scope",
        )

        self.assertEqual(summary["status"], "failed")
        self.assertFalse(summary["meets_threshold"])
        self.assertLess(
            summary["routing_accuracy_percent"],
            DEFAULT_MIN_ROUTING_ACCURACY,
        )

    def test_threshold_is_inclusive(self):
        queries = self.queries[:10]
        expected_by_question = {
            item["question"]: item["expected"] for item in queries
        }
        wrong_question = queries[0]["question"]

        def predictor(question):
            if question == wrong_question:
                return "out_of_scope"
            return expected_by_question[question]

        summary = evaluate_routing_queries(
            queries=queries,
            predictor=predictor,
            min_accuracy_percent=90.0,
        )

        self.assertEqual(summary["routing_accuracy_percent"], 90.0)
        self.assertTrue(summary["meets_threshold"])

    def test_rejects_invalid_threshold(self):
        with self.assertRaisesRegex(ValueError, "entre 0 y 100"):
            evaluate_routing_queries(
                queries=self.queries,
                predictor=self.expected_by_question.__getitem__,
                min_accuracy_percent=101.0,
            )

    def test_exit_code_reflects_acceptance_result(self):
        self.assertEqual(routing_test_exit_code({"meets_threshold": True}), 0)
        self.assertEqual(routing_test_exit_code({"meets_threshold": False}), 1)
        self.assertEqual(routing_test_exit_code({}), 1)


if __name__ == "__main__":
    unittest.main() 
