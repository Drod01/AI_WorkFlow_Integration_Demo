from fastapi import FastAPI
from typing import List
from .schemas import (
    AskRequest, AskResponse,
    ClassifyRequest, ClassifyResponse,
    SearchRequest, SearchResult,
    ExplainRequest, HealthResponse
)
from . import llm_service, vector_store, fraud_model, explainability

app = FastAPI(title="AI Workflow Integration Demo", version="1.0.0")

# Build components at startup
store = vector_store.VectorStore()
store.build()
model, meta = fraud_model.train_or_load()

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse()

@app.post("/semantic-search", response_model=List[SearchResult])
def semantic_search(req: SearchRequest):
    results = store.search(req.query, k=req.k, filter_ids=req.filter_ids)
    return [SearchResult(id=i, text=t, score=s) for i, t, s in results]

@app.post("/ask-llm", response_model=AskResponse)
def ask_llm(req: AskRequest):
    # Retrieve context
    results = store.search(req.query, k=4, filter_ids=req.context_ids)
    context_chunks = [r[1] for r in results]
    answer, provider = llm_service.generate_answer(req.query, context_chunks)

    # Simple "escalate to human" heuristic
    escalated = False
    risky_terms = ["wire money", "gift card", "password", "routing number", "account number"]
    if any(term in req.query.lower() for term in risky_terms):
        escalated = True
        answer += "\n\nNote: This appears sensitive. A human specialist will review."

    return AskResponse(answer=answer, provider=provider, used_context=context_chunks, escalated_to_human=escalated)

@app.post("/classify-fraud", response_model=ClassifyResponse)
def classify_fraud(req: ClassifyRequest):
    score = fraud_model.predict_proba(model, req.text)
    label = "fraud" if score >= req.threshold else "not_fraud"
    return ClassifyResponse(label=label, score=score)

@app.post("/explain")
def explain(req: ExplainRequest):
    return explainability.explain_pipeline(model, req.text)
