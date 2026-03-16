import os

from fastapi import FastAPI, Depends, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from services.ingestion import DocumentProcessor
from services.rag_engine import RAGEngine
from services.vector_store import VectorStoreManager
from schemas.requests import DiagnoseRequest
from utils.file_handlers import LocalStorageManager

app = FastAPI(
    title="Ghost-Shell API",
    description="RAG-Based Ghost-Shell API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "/data/raw_docs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

storage_manager = LocalStorageManager(base_upload_dir="/data/raw_docs")
doc_processor = DocumentProcessor()
vector_manager = VectorStoreManager()

rag_engine = RAGEngine(retriever=vector_manager.get_retriever(k=4))


def get_storage(): return storage_manager


def get_doc_processor(): return doc_processor


def get_vector_manager(): return vector_manager


def get_rag_engine(): return rag_engine


@app.get("/ping", tags=["Health"])
async def ping():
    """
    Health Check route to verify if the server is online.

    :return: A dictionary containing the health status of the API.
    """
    return {"status": "online",
            "statusCode": 200,
            "message": "pong",
            "version": app.version
            }


@app.post("/upload", tags=["Knowledge Base"])
async def upload_document(
        file: UploadFile = File(...),
        storage: LocalStorageManager = Depends(get_storage),
        processor: DocumentProcessor = Depends(get_doc_processor),
        v_manager: VectorStoreManager = Depends(get_vector_manager),
):
    """
    Receives a document via upload, processes it into chunks, and stores it in the vector store.

    :param file: The uploaded file containing technical documentation.
    :type file: fastapi.UploadFile
    :param processor: The document processor injected dependency.
    :type processor: DocumentProcessor
    :param v_manager: The vector store manager injected dependency.
    :type v_manager: VectorStoreManager
    :return: A dictionary with the upload status and the number of chunks inserted.
    """

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    try:
        file_path = await storage.save_file(file)

        chunks = processor.process_file(file_path)
        v_manager.add_document(chunks)

        return {
            "status": "success",
            "statusCode": 200,
            "chunks_inserted": len(chunks),
        }

    except ValueError as ve:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/diagnose", tags=["Troubleshooting"])
async def diagnose_error(
        request: DiagnoseRequest,
        engine: RAGEngine = Depends(get_rag_engine)
):
    """
    Receives an error log or symptom, queries the RAG engine, and returns a JSON diagnostic.

    :param request: The JSON payload containing the error log.
    :type request: DiagnoseRequest
    :param engine: The RAG Engine injected dependency.
    :type engine: RAGEngine
    :return: A structured JSON response with root cause and solution.
    """
    try:
        # Chama a inteligência do nosso SRE virtual
        diagnostic_result = engine.diagnose(request.error_log)

        return {
            "status": "success",
            "statusCode": 200,
            "data": diagnostic_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents", tags=["Knowledge Base"])
async def list_documents(v_manager: VectorStoreManager = Depends(get_vector_manager)):
    """
    Lists the metadata of all documents currently indexed in the vector store.

    :param v_manager: The vector store manager injected dependency.
    :type v_manager: VectorStoreManager
    :return: A dictionary containing the collection information.
    """

    try:
        info = v_manager.get_collection_info()
        return {
            "status": "success",
            "statusCode": 200,
            "data": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
