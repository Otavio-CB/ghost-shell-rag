from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import ChatOllama


class RAGEngine:
    """
    Orchestrates the Retrieval-Augmented Generation (RAG) pipeline.
    Connects the vector store retriever with the Ollama LLM to generate JSON diagnostics.
    """

    def __init__(self, retriever: VectorStoreRetriever):
        """
        Initializes the RAG Engine with a specific retriever and LLM configuration.

        :param retriever: The retriever instance from the VectorStoreManager.
        :type retriever: langchain_core.vectorstores.VectorStoreRetriever
        """
        self.retriever = retriever

        self.llm = ChatOllama(
            model="qwen2.5:3b",
            base_url="http://ollama:11434",
            temperature=0.1,
            format="json"
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Você é um Engenheiro de Confiabilidade de Sites (SRE) Sênior. "
             "Sua missão é diagnosticar logs de erro de infraestrutura e aplicações. "
             "Use OBRIGATORIAMENTE o Contexto Técnico fornecido abaixo para basear sua resposta. "
             "Identifique o idioma em que o usuário enviou o erro/pergunta e responda nesse mesmo idioma.\n\n"
             "Sua resposta DEVE ser estritamente um JSON válido com a seguinte estrutura:\n"
             "{{\n"
             "  \"root_cause\": \"A explicação técnica do problema baseado no contexto\",\n"
             "  \"solution\": \"O passo a passo acionável para resolver o problema\",\n"
             "  \"language_detected\": \"O idioma identificado no log/pergunta\"\n"
             "}}\n\n"
             "Contexto Técnico Extraído da Documentação:\n{context}"),
            ("human", "Analise este log/erro:\n{input}")
        ])

        self.output_parser = JsonOutputParser()

    def _format_docs(self, docs: list) -> str:
        """
        Helper method to format retrieved document chunks into a single string.

        :param docs: A list of retrieved document chunks.
        :type docs: list
        :return: A formatted string containing the combined page content.
        :rtype: str
        """
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    def diagnose(self, error_log: str) -> dict:
        """
        Executes the RAG pipeline: retrieves context, queries the LLM, 
        and parses the response into a structured JSON dictionary.

        :param error_log: The error stack trace or description provided by the user.
        :type error_log: str
        :return: A dictionary containing root_cause, solution, and language_detected.
        """
        retrieved_docs = self.retriever.invoke(error_log)
        context_string = self._format_docs(retrieved_docs)

        chain = self.prompt | self.llm | self.output_parser

        response = chain.invoke({
            "context": context_string,
            "input": error_log
        })

        response["sources_used"] = [
            doc.metadata.get("source", "unknown") for doc in retrieved_docs
        ]

        return response
