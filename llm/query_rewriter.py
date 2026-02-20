# query rewriting to improve retrieval
from typing import Optional

from config import LLM_MAX_TOKENS, LLM_TEMP

QUERY_REWRITE_PROMPT = """You are a search query rewriter. Given a user question, output a single improved search query that would help find relevant passages in a document collection. Keep it concise (1-2 sentences). Do not answer the question, only output the rewritten query.

User question: {query}

Rewritten search query:"""


def rewrite_query(
    query: str,
    client=None,
    max_tokens: int = 128,
    temperature: float = 0.1,
) -> str:
    if client is None:
        from .client import get_llm_client
        client = get_llm_client()
    prompt = QUERY_REWRITE_PROMPT.format(query=query)
    rewritten = client.generate(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
        repeat_penalty=1.05,
        stop=["User question:", "\n\n"],
    )
    return rewritten.strip() if rewritten else query
