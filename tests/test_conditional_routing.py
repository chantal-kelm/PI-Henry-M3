import unittest
from unittest.mock import patch

from langchain_core.runnables import RunnableBranch

from src import multi_agent_system


class ConditionalRoutingTests(unittest.TestCase):
    def setUp(self):
        self.select_domain_agent = (
            multi_agent_system.select_domain_agent.__wrapped__
        )

    def test_router_is_a_langchain_runnable_branch(self):
        router = multi_agent_system.get_conditional_agent_router()

        self.assertIsInstance(router, RunnableBranch)

    def test_each_destination_initializes_only_its_selected_agent(self):
        cases = [
            ("hr", "hr-agent", "get_hr_chain"),
            ("tech", "tech-agent", "get_tech_chain"),
            ("finance", "finance-agent", "get_finance_chain"),
        ]

        for destination, expected_agent, expected_factory in cases:
            with self.subTest(destination=destination):
                with (
                    patch.object(
                        multi_agent_system,
                        "get_hr_chain",
                        return_value="hr-agent",
                    ) as hr_factory,
                    patch.object(
                        multi_agent_system,
                        "get_tech_chain",
                        return_value="tech-agent",
                    ) as tech_factory,
                    patch.object(
                        multi_agent_system,
                        "get_finance_chain",
                        return_value="finance-agent",
                    ) as finance_factory,
                    patch.object(
                        multi_agent_system,
                        "get_langchain_callbacks",
                        return_value=[],
                    ),
                ):
                    selected = self.select_domain_agent(
                        destination,
                        "pregunta de prueba",
                    )

                self.assertEqual(selected, expected_agent)
                factories = {
                    "get_hr_chain": hr_factory,
                    "get_tech_chain": tech_factory,
                    "get_finance_chain": finance_factory,
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
                    patch.object(multi_agent_system, "get_hr_chain") as hr_factory,
                    patch.object(multi_agent_system, "get_tech_chain") as tech_factory,
                    patch.object(
                        multi_agent_system,
                        "get_finance_chain",
                    ) as finance_factory,
                    patch.object(
                        multi_agent_system,
                        "get_langchain_callbacks",
                        return_value=[],
                    ),
                ):
                    selected = self.select_domain_agent(
                        destination,
                        "pregunta de prueba",
                    )

                self.assertIsNone(selected)
                hr_factory.assert_not_called()
                tech_factory.assert_not_called()
                finance_factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
