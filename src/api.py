from fastapi import FastAPI
from pydantic import BaseModel
from .rag import answer

app = FastAPI(title="Mini RAG Bench", version="0.1.0")

class AskReq(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ask")
def ask(req: AskReq):
    return answer(req.question)
