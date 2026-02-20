# answer generation from retrieved context
from typing import List, Optional

from config import LLM_MAX_TOKENS, LLM_TEMP

ANSWER_PROMPT = """You are a careful assistant that answers only based on the provided context, which comes from private documents.

Instructions:
- Use the context below as the single source of truth.
- If the context does not contain enough information to answer, say exactly: "I don't know based on the indexed documents."
- Do not use any knowledge outside the context, even if you believe you know the answer.
- If the question is general (e.g. "What is time?") but the context is about something else, you must say "I don't know based on the indexed documents."
- If the question specifies an audience (e.g. beginner in ML), tailor your explanation to that level and prefer simple, clear language.

Context:
{context}

Question: {query}
Answer:"""


def build_context_from_chunks(chunk_texts: List[str], max_chars: int = 6000) -> str:
    parts = []
    total = 0
    for t in chunk_texts:
        if total + len(t) > max_chars:
            break
        parts.append(t)
        total += len(t)
    return "\n\n---\n\n".join(parts)


def generate_answer(
    query: str,
    context: str,
    client=None,
    max_tokens: int = LLM_MAX_TOKENS,
    temperature: float = LLM_TEMP,
) -> str:
    if not context or not context.strip():
        return "I don't know based on the indexed documents. The retrieved context was empty or not informative enough."
    if client is None:
        from .client import get_llm_client
        client = get_llm_client()
    prompt = ANSWER_PROMPT.format(context=context, query=query)
    return client.generate(
        prompt,
        max_tokens=max_tokens,
        temperature=0.1,
        top_p=0.9,
        repeat_penalty=1.1,
        stop=["Context:", "Question:", "\n\nQuestion:", "\n\nContext:"],
    )
