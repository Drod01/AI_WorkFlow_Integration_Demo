from typing import Dict
import shap
import numpy as np

def explain_pipeline(pipeline, text: str) -> Dict:
    # Use kernel explainer on the pipeline decision function
    # For demo speed, we use TF-IDF features on a small background.
    vec = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    # Build a small background sample
    background_texts = [
        "Please verify your account by sending us prepaid gift cards.",
        "Your paycheck has been deposited via ACH.",
        "Please reset your password using the official portal.",
        "Wire funds only to verified recipients."
    ]
    X_bg = vec.transform(background_texts)
    explainer = shap.LinearExplainer(clf, X_bg, feature_dependence="independent")
    X_text = vec.transform([text])
    shap_values = explainer.shap_values(X_text)

    # Map top features
    feature_names = vec.get_feature_names_out()
    contribs = shap_values[0].toarray().ravel()
    top_idx = np.argsort(np.abs(contribs))[::-1][:10]
    top_features = [{ "feature": feature_names[i], "contribution": float(contribs[i]) } for i in top_idx]

    return {
        "top_features": top_features
    }
