import os
import numpy as np
from typing import List, Tuple, Optional
import pandas as pd

# Optional deps
try:
    import faiss  # type: ignore
except Exception:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "seed_knowledge.csv")

class EmbeddingBackend:
    def __init__(self):
        self.backend = None
        self.model = None
        self.vectorizer = None

        if SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer(os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
                self.backend = "st"
            except Exception:
                pass
        if self.backend is None:
            self.vectorizer = TfidfVectorizer()
            self.backend = "tfidf"

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.backend == "st" and self.model is not None:
            return self.model.encode(texts, normalize_embeddings=True)
        # TFIDF fallback
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer()
        if not hasattr(self.vectorizer, "vocabulary_"):
            self.vectorizer.fit(texts)
        mat = self.vectorizer.transform(texts)
        # Return dense matrix
        return mat.toarray()

class VectorStore:
    def __init__(self):
        self.backend = EmbeddingBackend()
        self.ids: List[str] = []
        self.texts: List[str] = []
        self.tags: List[str] = []
        self.index = None  # FAISS index or numpy matrix

    def build(self, df: Optional[pd.DataFrame] = None):
        if df is None:
            df = pd.read_csv(DATA_PATH)
        self.ids = df["id"].astype(str).tolist()
        self.texts = df["text"].astype(str).tolist()
        self.tags = df.get("tag", pd.Series(["general"]*len(self.ids))).astype(str).tolist()
        embs = self.backend.encode(self.texts)
        if faiss is not None and self.backend.backend == "st":
            dim = embs.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(embs.astype("float32"))
        else:
            # fallback to cosine sim on numpy
            self.index = embs

    def search(self, query: str, k: int = 5, filter_ids: Optional[List[str]] = None) -> List[Tuple[str, str, float]]:
        if self.index is None:
            self.build()
        # filter
        mask = [True]*len(self.ids)
        if filter_ids:
            mask = [t in filter_ids for t in self.tags]
        texts = [t for t, m in zip(self.texts, mask) if m]
        ids = [i for i, m in zip(self.ids, mask) if m]

        # compute embeddings for filtered corpus + query
        embs_corpus = self.backend.encode(texts)
        emb_q = self.backend.encode([query])

        if faiss is not None and self.backend.backend == "st":
            # rebuild temp faiss for filtered set
            dim = embs_corpus.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(embs_corpus.astype("float32"))
            D, I = index.search(emb_q.astype("float32"), min(k, len(texts)))
            idxs = I[0]
            scores = D[0]
            results = [(ids[i], texts[i], float(scores[j])) for j, i in enumerate(idxs)]
        else:
            sims = cosine_similarity(emb_q, embs_corpus)[0]
            top_idx = sims.argsort()[::-1][:k]
            results = [(ids[i], texts[i], float(sims[i])) for i in top_idx]
        return results
