from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from services.rag_engine import RAGEngine


class TestRAGEngine:
    """
    Behavioral test suite for the RAGEngine service.
    Validates document formatting and the RAG pipeline execution using a FakeChatModel.
    """

    @pytest.fixture
    def mock_retriever(self):
        """
        Creates a mocked VectorStoreRetriever that returns a predefined list of Documents.

        :return: A mocked retriever instance.
        :rtype: unittest.mock.MagicMock
        """
        retriever = MagicMock()
        retriever.invoke.return_value = [
            Document(page_content="Log error 500 implies a server-side crash in the API.",
                     metadata={"source": "api_docs.md"}),
            Document(page_content="Ensure the AWS ECS task has enough memory.",
                     metadata={"source": "aws_troubleshooting.txt"})
        ]
        return retriever

    @pytest.fixture
    def engine(self, mock_retriever):
        """
        Instantiates the RAGEngine with the mocked retriever.

        :param mock_retriever: The mocked VectorStoreRetriever.
        :type mock_retriever: unittest.mock.MagicMock
        :return: A configured instance of the RAGEngine.
        :rtype: RAGEngine
        """
        return RAGEngine(retriever=mock_retriever)

    def test_format_docs_joins_content_correctly(self, engine, mock_retriever):
        """
        Verifies that the helper method correctly joins multiple document contents
        with the specified separator.

        :param engine: The RAGEngine fixture instance.
        :type engine: RAGEngine
        :param mock_retriever: The mocked VectorStoreRetriever.
        :type mock_retriever: unittest.mock.MagicMock
        """
        docs = mock_retriever.invoke("dummy query")

        formatted_string = engine._format_docs(docs)

        expected_separator = "\n\n---\n\n"
        assert expected_separator in formatted_string
        assert "Log error 500 implies" in formatted_string
        assert "Ensure the AWS ECS" in formatted_string

    def test_diagnose_success_returns_json_with_sources(self, engine):
        """
        Validates the complete execution of the RAG pipeline. Replaces the internal 
        ChatOllama with a FakeListChatModel to simulate a JSON response, ensuring 
        the JSON Output Parser and source appending logic work flawlessly.

        :param engine: The RAGEngine fixture instance.
        :type engine: RAGEngine
        """
        error_input = "Getting 500 error on ECS deployment"

        fake_llm = FakeListChatModel(
            responses=['{"root_cause": "Out of memory", "solution": "Increase task size", "language_detected": "en"}']
        )

        engine.llm = fake_llm

        result = engine.diagnose(error_input)

        assert isinstance(result, dict)
        assert result["root_cause"] == "Out of memory"
        assert result["solution"] == "Increase task size"
        assert result["language_detected"] == "en"

        assert "sources_used" in result
        assert len(result["sources_used"]) == 2
        assert "api_docs.md" in result["sources_used"]
        assert "aws_troubleshooting.txt" in result["sources_used"]

    def test_diagnose_handles_unknown_sources(self, engine):
        """
        Ensures that documents without a 'source' key in their metadata default to 'unknown'
        to prevent KeyErrors during dictionary generation.

        :param engine: The RAGEngine fixture instance.
        :type engine: RAGEngine
        """
        engine.retriever.invoke.return_value = [
            Document(page_content="Mystery text.", metadata={})
        ]

        fake_llm = FakeListChatModel(
            responses=['{"root_cause": "Unknown", "solution": "N/A", "language_detected": "en"}']
        )
        engine.llm = fake_llm

        result = engine.diagnose("Any error")

        assert "sources_used" in result
        assert result["sources_used"] == ["unknown"]
