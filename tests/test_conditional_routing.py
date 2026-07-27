import unittest
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.runnables import RunnableBranch, RunnableLambda

from src import multi_agent_system
from src.agents.rag_chain import build_domain_rag_chain


class ConditionalRoutingTests(unittest.TestCase):
    def test_router_is_a_langchain_runnable_branch(self):
        router = multi_agent_system.get_conditional_agent_router()

        self.assertIsInstance(router, RunnableBranch)

    def test_full_orchestrator_classifies_and_executes_selected_rag_branch(self):
        fake_hr_agent = RunnableLambda(
            lambda route_input: {
                **route_input,
                "context": "política de vacaciones",
                "doc_sources": ["doc_hr.txt"],
                "response": "respuesta de HR",
            }
        )

        with (
            patch.object(
                multi_agent_system,
                "get_orchestrator_chain",
                return_value=RunnableLambda(lambda _route_input: "hr"),
            ),
            patch.object(
                multi_agent_system,
                "get_hr_rag_chain",
                return_value=fake_hr_agent,
            ) as hr_factory,
            patch.object(
                multi_agent_system,
                "get_tech_rag_chain",
            ) as tech_factory,
            patch.object(
                multi_agent_system,
                "get_finance_rag_chain",
            ) as finance_factory,
        ):
            result = multi_agent_system.get_multi_agent_orchestrator().invoke(
                {"question": "¿Cuántos días de vacaciones tengo?"}
            )

        self.assertEqual(result["destination"], "hr")
        self.assertEqual(result["context"], "política de vacaciones")
        self.assertEqual(result["response"], "respuesta de HR")
        hr_factory.assert_called_once_with()
        tech_factory.assert_not_called()
        finance_factory.assert_not_called()

    def test_domain_rag_chain_retrieves_once_and_generates_inside_the_runnable(self):
        retrieval_calls = []
        retriever = RunnableLambda(
            lambda question: (
                retrieval_calls.append(question)
                or [
                    Document(
                        page_content="La licencia tiene una duración de 90 días.",
                        metadata={"source": "doc_hr.txt"},
                    )
                ]
            )
        )
        answer_chain = RunnableLambda(
            lambda agent_input: f"Respuesta basada en: {agent_input['context']}"
        )
        rag_agent = build_domain_rag_chain(
            answer_chain=answer_chain,
            retriever=retriever,
            domain="hr",
        )

        result = rag_agent.invoke(
            {
                "question": "¿Cuánto dura la licencia?",
                "destination": "hr",
            }
        )

        self.assertEqual(retrieval_calls, ["¿Cuánto dura la licencia?"])
        self.assertEqual(result["doc_sources"], ["doc_hr.txt"])
        self.assertIn("90 días", result["context"])
        self.assertIn("90 días", result["response"])

    def test_each_destination_executes_only_its_selected_complete_rag_agent(self):
        cases = [
            ("hr", "get_hr_rag_chain"),
            ("tech", "get_tech_rag_chain"),
            ("finance", "get_finance_rag_chain"),
        ]

        for destination, expected_factory in cases:
            with self.subTest(destination=destination):
                def build_fake_rag_agent(agent_destination):
                    return RunnableLambda(
                        lambda route_input: {
                            "question": route_input["question"],
                            "destination": agent_destination,
                            "context": f"contexto-{agent_destination}",
                            "doc_sources": [f"{agent_destination}.txt"],
                            "response": f"respuesta-{agent_destination}",
                        }
                    )

                with (
                    patch.object(
                        multi_agent_system,
                        "get_hr_rag_chain",
                        return_value=build_fake_rag_agent("hr"),
                    ) as hr_factory,
                    patch.object(
                        multi_agent_system,
                        "get_tech_rag_chain",
                        return_value=build_fake_rag_agent("tech"),
                    ) as tech_factory,
                    patch.object(
                        multi_agent_system,
                        "get_finance_rag_chain",
                        return_value=build_fake_rag_agent("finance"),
                    ) as finance_factory,
                ):
                    result = (
                        multi_agent_system.get_conditional_agent_router().invoke(
                            {
                                "destination": destination,
                                "question": "pregunta de prueba",
                            }
                        )
                    )

                self.assertEqual(result["destination"], destination)
                self.assertEqual(result["question"], "pregunta de prueba")
                self.assertEqual(result["context"], f"contexto-{destination}")
                self.assertEqual(result["response"], f"respuesta-{destination}")
                factories = {
                    "get_hr_rag_chain": hr_factory,
                    "get_tech_rag_chain": tech_factory,
                    "get_finance_rag_chain": finance_factory,
                }
                for factory_name, factory in factories.items():
                    self.assertEqual(
                        factory.call_count,
                        1 if factory_name == expected_factory else 0,
                    )

    def test_out_of_scope_and_unknown_destinations_do_not_initialize_agents(self):
        for destination in ("out_of_scope", "unknown"):
            with self.subTest(destination=destination):
                with (
                    patch.object(
                        multi_agent_system,
                        "get_hr_rag_chain",
                    ) as hr_factory,
                    patch.object(
                        multi_agent_system,
                        "get_tech_rag_chain",
                    ) as tech_factory,
                    patch.object(
                        multi_agent_system,
                        "get_finance_rag_chain",
                    ) as finance_factory,
                ):
                    result = (
                        multi_agent_system.get_conditional_agent_router().invoke(
                            {
                                "destination": destination,
                                "question": "pregunta de prueba",
                            }
                        )
                    )

                self.assertEqual(result["destination"], "out_of_scope")
                self.assertEqual(result["context"], "")
                self.assertEqual(result["doc_sources"], [])
                hr_factory.assert_not_called()
                tech_factory.assert_not_called()
                finance_factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
