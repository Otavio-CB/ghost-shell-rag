import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def process_file(self, file_path: str):
        """
        Processes a single file, loads its content, and returns the split document chunks.

        :param file_path: The absolute or relative path to the file to be processed.
        :type file_path: str
        :return: A list of processed document chunks.
        """

        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext in [".md", ".txt"]:
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        documents = loader.load()
        return self.text_splitter.split_documents(documents)
