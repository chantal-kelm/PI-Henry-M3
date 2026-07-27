import unittest
from unittest.mock import patch

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.agents import finance_agent, hr_agent, tech_agent


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0]


class AgentCachingTests(unittest.TestCase):
    def test_each_domain_builds_its_index_once_per_cache_lifecycle(self):
        cases = [
            (hr_agent, "get_hr_chain", 57),
            (tech_agent, "get_tech_chain", 53),
            (finance_agent, "get_finance_chain", 70),
        ]

        for module, factory_name, expected_chunks in cases:
            with self.subTest(domain=module.__name__):
                factory = getattr(module, factory_name)
                factory.cache_clear()

                with (
                    patch.object(
                        module,
                        "ChatOpenAI",
                        side_effect=lambda **_kwargs: FakeListChatModel(
                            responses=["ok"]
                        ),
                    ) as chat_openai,
                    patch.object(
                        module,
                        "OpenAIEmbeddings",
                        side_effect=lambda **_kwargs: FakeEmbeddings(),
                    ) as embeddings,
                ):
                    first = factory()
                    second = factory()

                    self.assertIs(first, second)
                    self.assertEqual(chat_openai.call_count, 1)
                    self.assertEqual(embeddings.call_count, 1)
                    self.assertEqual(
                        len(first[1].vectorstore.store),
                        expected_chunks,
                    )

                    cache_info = factory.cache_info()
                    self.assertEqual(cache_info.misses, 1)
                    self.assertEqual(cache_info.hits, 1)
                    self.assertEqual(cache_info.currsize, 1)

                    factory.cache_clear()
                    rebuilt = factory()

                    self.assertIsNot(rebuilt, first)
                    self.assertEqual(chat_openai.call_count, 2)
                    self.assertEqual(embeddings.call_count, 2)

                factory.cache_clear()


if __name__ == "__main__":
    unittest.main()
