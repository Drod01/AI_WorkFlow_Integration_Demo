import os
import pickle
import pandas as pd
from typing import Tuple
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "fraud_classifier.pkl")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sample_fraud.csv")

def _build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000))
    ])

def train_or_load() -> Tuple[Pipeline, dict]:
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        return model, {"status": "loaded"}

    df = pd.read_csv(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])
    pipe = _build_pipeline()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipe, f)
    return pipe, {"status": "trained", "report": report}

def predict_proba(model: Pipeline, text: str) -> float:
    # Return probability of class "fraud"
    proba = model.predict_proba([text])[0]
    # Assumes label order is ["fraud","not_fraud"] or ["not_fraud","fraud"]
    if "fraud" in model.classes_:
        fraud_index = list(model.classes_).index("fraud")
        return float(proba[fraud_index])
    # Fallback: assume positive class is index 1
    return float(proba[-1])
