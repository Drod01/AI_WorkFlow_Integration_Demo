# AI Workflow Integration Demo

Project Description:
- Build and support **AI/ML components**
- Integrate **LLMs** into enterprise-style **REST APIs**
- Apply **prompt engineering** and **chaining**
- Combine **human + system + AI** workflow orchestration
- Use **embeddings** and a **vector database** for semantic retrieval
- Provide **explainability** for ML predictions
- Package with **Docker** and a starter **CI/CD (GitHub Actions)**



---

## ✨ What This Project Demonstrates

- **FastAPI microservice** with endpoints:
  - `POST /ask-llm` – LLM-backed financial FAQ & assistant
  - `POST /classify-fraud` – Binary fraud text classifier (scikit-learn)
  - `POST /semantic-search` – FAISS-backed similarity search over past Q&A
  - `POST /explain` – SHAP explanation for a given classifier prediction
  - `GET /health` – Healthcheck
- **Prompt chaining** – System → Guardrails → User → Post-processing
- **Vector DB (FAISS)** – Flexible embedding layer with **SentenceTransformers** fallback to **TF‑IDF** when transformers aren't available
- **Human-in-the-loop routing** – Route suspicious/high-risk outputs to human review
- **Dockerized** – One-line build & run
- **CI/CD** – Lint + tests on push via GitHub Actions
- **Clear codebase** – Well-structured modules, pydantic schemas, unit tests

---

## 🧱 Project Structure

```
ai_workflow_integration_demo/
│── app/
│   ├── main.py
│   ├── schemas.py
│   ├── llm_service.py
│   ├── fraud_model.py
│   ├── vector_store.py
│   ├── explainability.py
│   └── workflow.py
│
│── data/
│   ├── sample_fraud.csv
│   └── seed_knowledge.csv
│
│── models/
│   └── fraud_classifier.pkl
│
│── tests/
│   └── test_api.py
│
│── .github/workflows/ci.yml
│── Dockerfile
│── requirements.txt
│── README.md
```

---

## 🚀 Quickstart

### 1) Local (Python 3.10+)
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# (Optional) Use OpenAI - set your key:
# export OPENAI_API_KEY=sk-...

# Run API
uvicorn app.main:app --reload

# Open interactive docs:
# http://127.0.0.1:8000/docs
```

> **Note**: If `sentence-transformers` model download is blocked, the service will automatically fall back to a local TF‑IDF embedding backend.

### 2) Docker
```bash
docker build -t ai-workflow-demo .
docker run -p 8000:8000 --env OPENAI_API_KEY=$OPENAI_API_KEY ai-workflow-demo
```

---

## 🔌 REST Endpoints

- `POST /ask-llm`
  - Body: `{ "query": "How do I increase my credit limit?", "context_ids": ["faq","fraud"] }`
  - Returns assistant message backed by LLM (OpenAI/Transformers) with safe guardrails and optional retrieval-augmented context.

- `POST /classify-fraud`
  - Body: `{ "text": "Wire money to this account now...", "threshold": 0.7 }`
  - Returns `{ "label": "fraud", "score": 0.92 }`

- `POST /semantic-search`
  - Body: `{ "query": "ACH transfer limits", "k": 5 }`
  - Returns top-k similar seed knowledge items.

- `POST /explain`
  - Body: `{ "text": "Please send gift cards to resolve your account issue." }`
  - Returns SHAP explanation values for the classifier.

- `GET /health`
  - Returns `{"status":"ok"}`

---

## 🧠 LLM Integration Strategy

The `llm_service` module attempts providers in this order:

1. **OpenAI API** (if `OPENAI_API_KEY` env is set)
2. **Hugging Face Transformers** (if available locally)
3. **Rule-based fallback** – Safe, deterministic responses for offline demos

Prompt **chaining** injects:
- A **system prompt** (policies & style)
- **Guardrails** (refuse secrets, PII requests, etc.)
- **RAG context** (from FAISS/TF‑IDF store)
- **User message**
- **Post‑processor** (shorten, add disclaimers for risk)

---

## ⚖️ Explainability

Explainability is provided via **SHAP** for the scikit-learn pipeline (TF‑IDF + LogisticRegression). See `app/explainability.py` and the `/explain` endpoint.

---

## 🧪 Tests

Run tests:
```bash
pytest -q
```

---

## 🛠️ Requirements / Software Used

- Python 3.10+
- FastAPI, Uvicorn
- scikit-learn, pandas, numpy
- sentence-transformers (optional), faiss-cpu (optional)
- shap (explainability)
- pydantic
- Docker
- GitHub Actions (CI)

See `requirements.txt` for full list.

---

## 🔧 CI/CD (GitHub Actions)

- Lints and runs tests on every push/pull request
- Easy to extend for build/publish steps

---

## 🧭 Roadmap Ideas

- Swap local FAISS with managed vector DB
- Add OAuth/JWT authentication
- Add Kafka/NATS for event-driven flows
- Container orchestration with Kubernetes
- Model registry via MLflow

---
