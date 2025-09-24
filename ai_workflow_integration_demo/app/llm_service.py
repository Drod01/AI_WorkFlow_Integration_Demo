import os
from typing import List, Tuple
from . import workflow

# Optional providers
try:
    import openai  # type: ignore
except Exception:
    openai = None

try:
    from transformers import pipeline  # type: ignore
except Exception:
    pipeline = None

PROVIDER_OPENAI = "openai"
PROVIDER_HF = "huggingface"
PROVIDER_RULES = "rule-based"

def _call_openai(messages: List[dict]) -> str:
    if openai is None:
        raise RuntimeError("openai library not available")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    chat = client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.2)
    return chat.choices[0].message.content or ""

def _call_hf(messages: List[dict]) -> str:
    if pipeline is None:
        raise RuntimeError("transformers not available")
    # Use a small instruction-tuned model if present; otherwise, a general model.
    generator = pipeline("text-generation", model=os.getenv("HF_MODEL", "distilgpt2"), max_new_tokens=256)
    # Naive concat for demo
    prompt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages[-4:])
    out = generator(prompt)[0]["generated_text"]
    return out.split("USER:")[-1].strip() if "USER:" in out else out.strip()

def _call_rules(messages: List[dict]) -> str:
    # Very small safety-aware template
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    if any(k in last_user.lower() for k in ["password", "social security", "ssn", "api key", "secret"]):
        return "I can’t help with sensitive or personal data. For account-specific help, please contact support."
    if "credit limit" in last_user.lower():
        return ("You can request a credit limit increase via your account portal or by contacting support. "
                "Lenders typically review payment history, utilization, and income. Avoid hard inquiries if possible.")
    if "ach" in last_user.lower() or "transfer" in last_user.lower():
        return ("ACH transfers are electronic bank-to-bank transfers. Same-day ACH has earlier cutoffs and fees may apply. "
                "Verify routing/account numbers and beware of phishing.")
    return ("Here’s a general answer based on limited context: "
            "Please clarify details, and I can tailor guidance. If this concerns account security, involve a human specialist.")

def generate_answer(user_query: str, context_chunks: List[str]) -> Tuple[str, str]:
    messages = workflow.build_prompt(user_query, context_chunks)

    # Try providers in order: OpenAI -> HF -> Rules
    try:
        if os.getenv("OPENAI_API_KEY"):
            return workflow.postprocess(_call_openai(messages)), PROVIDER_OPENAI
    except Exception:
        pass

    try:
        return workflow.postprocess(_call_hf(messages)), PROVIDER_HF
    except Exception:
        pass

    return workflow.postprocess(_call_rules(messages)), PROVIDER_RULES
