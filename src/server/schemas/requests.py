from pydantic import BaseModel

class DiagnoseRequest(BaseModel):
    """
    Schema for the incoming error diagnostic request.
    Validates that the client sends a JSON with the 'error_log' string.
    """
    error_log: str