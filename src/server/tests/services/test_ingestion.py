from unittest.mock import patch, MagicMock

import pytest
from langchain_core.documents import Document
from services.ingestion import DocumentProcessor


class TestDocumentProcessor:
    """
    Behavioral test suite for the DocumentProcessor service.
    Validates text splitting logic and file format routing using the AAA pattern.
    """

    @pytest.fixture
    def processor(self):
        """
        Instantiates the DocumentProcessor with a small chunk size to force
        the splitting behavior for easier testing.

        :return: A configured instance of the DocumentProcessor.
        :rtype: DocumentProcessor
        """
        return DocumentProcessor(chunk_size=50, chunk_overlap=10)

    @pytest.mark.parametrize("extension", [".txt", ".md"])
    def test_process_file_text_based_documents(self, processor, tmp_path, extension):
        """
        Validates that text-based files (.txt, .md) are correctly loaded
        and split into smaller chunks according to the configured parameters.

        :param processor: The DocumentProcessor fixture instance.
        :type processor: DocumentProcessor
        :param tmp_path: The pytest fixture providing a temporary directory path.
        :type tmp_path: pathlib.Path
        :param extension: The file extension being tested (e.g., '.txt', '.md').
        :type extension: str
        """
        target_file = tmp_path / f"sample_document{extension}"
        content = "This is a sufficiently long sentence that should definitely be split into multiple smaller chunks by the recursive text splitter."
        target_file.write_text(content, encoding="utf-8")

        chunks = processor.process_file(str(target_file))

        assert len(chunks) > 1
        assert chunks[0].metadata["source"] == str(target_file)
        assert "This is a sufficiently" in chunks[0].page_content

    @patch("services.ingestion.PyPDFLoader")
    def test_process_file_pdf_document(self, mock_pdf_loader_class, processor, tmp_path):
        """
        Validates the PDF routing and splitting logic by mocking the PyPDFLoader
        to avoid generating heavy binary files during testing.

        :param mock_pdf_loader_class: The mocked PyPDFLoader class from unittest.mock.
        :type mock_pdf_loader_class: MagicMock
        :param processor: The DocumentProcessor fixture instance.
        :type processor: DocumentProcessor
        :param tmp_path: The pytest fixture providing a temporary directory path.
        :type tmp_path: pathlib.Path
        """
        pdf_file = tmp_path / "fake_manual.pdf"
        pdf_file.touch()

        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = [
            Document(page_content="Fake PDF content that represents a parsed page.", metadata={"source": str(pdf_file)})
        ]
        mock_pdf_loader_class.return_value = mock_loader_instance

        chunks = processor.process_file(str(pdf_file))

        mock_pdf_loader_class.assert_called_once_with(str(pdf_file))
        assert len(chunks) > 0
        assert chunks[0].metadata["source"] == str(pdf_file)

    def test_process_file_unsupported_extension_raises_error(self, processor, tmp_path):
        """
        Verifies that passing an unsupported file extension raises a ValueError.

        :param processor: The DocumentProcessor fixture instance.
        :type processor: DocumentProcessor
        :param tmp_path: The pytest fixture providing a temporary directory path.
        :type tmp_path: pathlib.Path
        """
        invalid_file = tmp_path / "image.png"
        invalid_file.touch()

        with pytest.raises(ValueError, match="Unsupported file type: .png"):
            processor.process_file(str(invalid_file))
