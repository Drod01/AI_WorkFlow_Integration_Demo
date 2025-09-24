from typing import List, Dict

SYSTEM_PROMPT = (
    "You are a helpful, safety-aware banking assistant. "
    "Be concise and cite relevant policy when refusing unsafe requests. "
    "Never give out secrets, API keys, or personal data. "
)

GUARDRAILS = [
    "If the user asks for personal data or secrets → refuse.",
    "If the user asks for legal, medical, or tax advice → add a disclaimer and suggest contacting a professional.",
    "If uncertain or confidence is low → recommend speaking with a human specialist."
]

def build_prompt(user_query: str, context_chunks: List[str]) -> List[Dict[str, str]]:
    context_block = "\n\n".join(f"- {c}" for c in context_chunks) if context_chunks else "No extra context."
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": "Guardrails:\n" + "\n".join(f"* {g}" for g in GUARDRAILS)},
        {"role": "system", "content": f"Context to consider:\n{context_block}"},
        {"role": "user", "content": user_query},
    ]
    return messages

def postprocess(answer: str) -> str:
    # Simple post-processor to keep answers crisp
    if len(answer) > 1200:
        answer = answer[:1200] + "..."
    return answer.strip()
