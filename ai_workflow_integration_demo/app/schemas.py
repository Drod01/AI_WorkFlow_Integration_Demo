from pydantic import BaseModel, Field
from typing import List, Optional

class AskRequest(BaseModel):
    query: str = Field(..., description="User query for the assistant.")
    context_ids: Optional[List[str]] = Field(default=None, description="Optional tags to filter retrieval.")

class AskResponse(BaseModel):
    answer: str
    provider: str
    used_context: Optional[List[str]] = None
    escalated_to_human: bool = False

class ClassifyRequest(BaseModel):
    text: str
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)

class ClassifyResponse(BaseModel):
    label: str
    score: float

class SearchRequest(BaseModel):
    query: str
    k: int = 5
    filter_ids: Optional[List[str]] = None

class SearchResult(BaseModel):
    id: str
    text: str
    score: float

class ExplainRequest(BaseModel):
    text: str

class HealthResponse(BaseModel):
    status: str = "ok"
