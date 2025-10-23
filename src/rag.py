import os
from .retriever import TfidfRetriever
from .prompts import SYSTEM, build_user_prompt
from .llm import chat

TOP_K = int(os.getenv("TOP_K", "4"))
MAX_CTX = int(os.getenv("MAX_CTX_CHARS", "3000"))
RERANK = os.getenv("RERANK", "0") == "1"

_ret = None
def get_ret():
    global _ret
    if _ret is None:
        _ret = TfidfRetriever("artifacts/tfidf.pkl")
    return _ret

def answer(question: str):
    try:
        r = get_ret().topk(question, k=TOP_K, rerank=RERANK)
    except FileNotFoundError:
        return {"answer": "I don't have enough information.", "sources": []}
    if not r:
        return {"answer": "I don't have enough information.", "sources": []}
    up = build_user_prompt(question, r, max_chars=MAX_CTX)
    out = chat(SYSTEM, up)
    cites = [{"id": c["id"], "start": c["start"], "end": c["end"], "score": c["score"]} for c in r]
    return {"answer": out, "sources": cites}
