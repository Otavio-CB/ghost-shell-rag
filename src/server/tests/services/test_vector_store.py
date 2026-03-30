import hashlib
from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from services.vector_store import VectorStoreManager


class TestVectorStoreManager:
    """
    Behavioral test suite for the VectorStoreManager service.
    Validates deterministic ID generation, retriever configuration, and collection info parsing.
    External dependencies (ChromaDB and Ollama) are entirely mocked.
    """

    @pytest.fixture
    @patch("services.vector_store.Chroma")
    @patch("services.vector_store.chromadb.HttpClient")
    @patch("services.vector_store.OllamaEmbeddings")
    def manager(self, mock_embeddings, mock_http_client, mock_chroma_class):
        """
        Instantiates the VectorStoreManager while mocking its external connections.

        :param mock_embeddings: Mock for OllamaEmbeddings.
        :type mock_embeddings: unittest.mock.MagicMock
        :param mock_http_client: Mock for chromadb.HttpClient.
        :type mock_http_client: unittest.mock.MagicMock
        :param mock_chroma_class: Mock for the LangChain Chroma wrapper class.
        :type mock_chroma_class: unittest.mock.MagicMock
        :return: A configured instance of the VectorStoreManager with mocked internals.
        :rtype: VectorStoreManager
        """
        manager_instance = VectorStoreManager(collection_name="test_collection")

        manager_instance._mock_chroma_instance = mock_chroma_class.return_value
        return manager_instance

    def test_add_document_generates_deterministic_ids(self, manager):
        """
        Validates that add_document generates the correct SHA-256 hashes based on
        the content and the source metadata, preventing duplicate insertions.

        :param manager: The VectorStoreManager fixture instance.
        :type manager: VectorStoreManager
        """
        chunk1 = Document(page_content="print('hello')", metadata={"source": "script.py"})
        chunk2 = Document(page_content="def func(): pass", metadata={"source": "utils.py"})
        chunks = [chunk1, chunk2]

        expected_hash1 = hashlib.sha256("script.py::print('hello')".encode("utf-8")).hexdigest()
        expected_hash2 = hashlib.sha256("utils.py::def func(): pass".encode("utf-8")).hexdigest()

        manager.add_document(chunks)

        manager._mock_chroma_instance.add_documents.assert_called_once_with(
            documents=chunks,
            ids=[expected_hash1, expected_hash2]
        )

    def test_add_document_handles_missing_source(self, manager):
        """
        Ensures that chunks missing the 'source' metadata default to 'unknown' 
        without breaking the hashing logic.

        :param manager: The VectorStoreManager fixture instance.
        :type manager: VectorStoreManager
        """
        chunk = Document(page_content="Orphaned text.", metadata={})

        expected_hash = hashlib.sha256("unknown::Orphaned text.".encode("utf-8")).hexdigest()

        manager.add_document([chunk])

        manager._mock_chroma_instance.add_documents.assert_called_once_with(
            documents=[chunk],
            ids=[expected_hash]
        )

    def test_get_retriever_passes_correct_kwargs(self, manager):
        """
        Verifies that the get_retriever method passes the desired 'k' parameter 
        down to the Chroma as_retriever method.

        :param manager: The VectorStoreManager fixture instance.
        :type manager: VectorStoreManager
        """
        k_value = 7

        manager.get_retriever(k=k_value)

        manager._mock_chroma_instance.as_retriever.assert_called_once_with(
            search_kwargs={"k": k_value}
        )

    def test_get_collection_info_returns_correct_dictionary(self, manager):
        """
        Validates that get_collection_info properly parses the raw dictionary 
        returned by Chroma into the structured expected output.

        :param manager: The VectorStoreManager fixture instance.
        :type manager: VectorStoreManager
        """
        mock_raw_data = {
            "ids": ["id_1", "id_2"],
            "metadatas": [{"source": "doc1.txt"}, {"source": "doc2.txt"}],
            "documents": ["text1", "text2"]
        }
        manager._mock_chroma_instance.get.return_value = mock_raw_data

        info = manager.get_collection_info()

        assert isinstance(info, dict)
        assert info["total_chunks"] == 2
        assert len(info["metadatas"]) == 2
        assert info["metadatas"][0]["source"] == "doc1.txt"
