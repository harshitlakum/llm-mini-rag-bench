# llm-mini-rag-bench — Tiny, RAG + Eval (no training, no vector DB)

A minimal Retrieval-Augmented Generation (RAG) template designed to signal **research literacy** without any heavy infra.  
It shows: **clean TF-IDF retrieval**, **tight prompts**, **span-level citations**, a **tiny FastAPI API**, **deterministic caching**, **latency/token/cost logging**, and a **baseline vs RAG eval**.  
Runs fully **free** in MOCK mode; switch to a real API key later if you want.

---

## Why this repo exists

Most “LLM” repos either (1) train models or (2) bolt on big vector DBs. That’s overkill for a candidate portfolio or a research demo.  
This repo intentionally focuses on the three skills that matter in interviews and early research:

1. **Retrieval hygiene** — getting the *right* context in, with a clear budget.  
2. **Prompting discipline** — answer only from context, cite your sources.  
3. **Evaluation mindset** — measure retrieval and *answers*, not vibes.

It’s a compact, auditable demo you can run in minutes and discuss confidently.

---

## What’s inside

```

llm-mini-rag-bench/
├─ README.md
├─ requirements.txt
├─ .env.example
├─ data/
│  └─ docs/                 # your .txt/.md knowledge files
├─ artifacts/               # built index, logs, (optional) cache
├─ src/
│  ├─ ingest.py             # builds TF-IDF index over paragraph chunks (pickle)
│  ├─ retriever.py          # load index, top-k search (+ optional LSA re-rank)
│  ├─ prompts.py            # strict system prompt + context budgeting
│  ├─ llm.py                # OpenAI wrapper + MOCK mode + logging + cache
│  ├─ rag.py                # retrieve → compose prompt → call LLM → cite spans
│  └─ api.py                # FastAPI: GET /health, POST /ask
├─ eval/
│  ├─ eval.jsonl            # {"q":..., "ref":..., "keywords":[...]}
│  └─ run_eval.py           # hit@5 + keyword F1 (baseline vs RAG)
└─ tests/
├─ conftest.py
├─ test_retriever.py
└─ test_prompt.py

````

---

## Key behaviors

- **Chunked ingest + span citations**  
  Each doc is split into paragraph-ish chunks. Answers include inline markers like **[1]**, **[2]**.  
  The API also returns exact spans: `{"id":"doc.md","start":12,"end":180,"score":0.42}`.

- **Prompting discipline**  
  System prompt enforces *context-only* answers. If missing, the model must say:  
  > “I don’t have enough information.”

- **Logging & cost awareness**  
  Every call logs `latency_s`, token counts, and **estimated cost** to `artifacts/llm_log.csv`.

- **Deterministic cache (real API mode)**  
  Hash of `(model | system | prompt)` → `artifacts/cache.jsonl` to avoid repeat charges.

- **Optional LSA re-rank**  
  Lightweight SVD over the TF-IDF candidate set (no external services) to refine top-k.

---

## Setup

```bash
pip install -r requirements.txt
````

Put a few `.txt/.md` files into `data/docs/`, then build the index:

```bash
python -m src.ingest
# -> creates artifacts/tfidf.pkl
```

---

## Run it free (MOCK mode)

MOCK mode returns a short, context-flavored answer and still exercises retrieval, prompting, API, and logging — with **zero network calls**.

Set in your shell:

```bash
export MOCK=1
```

Or in `.env` (recommended):

```
MOCK=1
```

Start the API:

```bash
uvicorn src.api:app --reload
```

Try it:

```bash
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is TF-IDF used for?"}'
```

You’ll also see rows appended to `artifacts/llm_log.csv` (latency, tokens=0, cost=0).

---

## Real API (optional)

If/when you want real LLM answers:

1. Create `.env`:

```
OPENAI_API_KEY=sk-...         # your real key
MODEL_NAME=gpt-4o-mini
TOP_K=4
MAX_CTX_CHARS=3000
CACHE=1
MOCK=0
RERANK=0
```

2. Run again (`MOCK=0`). You’ll get proper answers; logs will include tokens and estimated cost.

---

## Programmatic use

```python
from src.rag import answer
print(answer("What is TF-IDF used for?"))
# {
#   "answer": "… [1]",
#   "sources": [{"id":"intro.md","start":0,"end":122,"score":0.31}, ...]
# }
```

---

## Evaluation (baseline vs RAG)

This repo includes a minimal harness that compares:

* **hit@5** — do retrieved chunks contain the gold keywords?
* **Keyword F1** — does the answer include the key terms?

Run:

```bash
python -m eval.run_eval
# -> hit@5=0.667 | F1_rag=0.067 | F1_base=0.067   (numbers depend on your data and MOCK/real)
```

Expectation (with a real model): **F1_rag > F1_base**.

---

## Config (env vars)

```
MODEL_NAME=gpt-4o-mini
TOP_K=4
MAX_CTX_CHARS=3000
CACHE=1        # enable deterministic cache in real API mode
MOCK=1         # 1 = free, no network; 0 = real API
RERANK=0       # 1 = enable LSA re-rank over TF-IDF candidates
```

---

## Tests

```bash
pytest -q
# 2 passed
```


---

## Troubleshooting

* **`python-dotenv` + Python 3.13**: if `load_dotenv()` is flaky in one-liners, set env in the shell (`export MOCK=1`) or run code from the repo root.
* **No docs ingested**: `python -m src.ingest` must run before queries.
* **Empty answers**: ensure your question overlaps with the content in `data/docs/`.

---

## License

MIT


