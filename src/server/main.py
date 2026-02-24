from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

@app.get("/ping", tags=["Health"])
async def ping():
    """
    Rota de Health Check para verificar se o servidor está online
    """
    return {"status": "online",
            "statusCode": 200,
            "message": "pong",
            "version": app.version
            }