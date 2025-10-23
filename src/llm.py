import os, time, re
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
from .logger import log_row, cache_get, cache_put, key_hash

_client = OpenAI()
_PRICE = {"gpt-4o-mini": {"in": 0.00015, "out": 0.00060}}  # USD per 1k tokens

def _tokens_from_usage(u):
    if not u:
        return 0, 0
    pt = getattr(u, "prompt_tokens", None)
    ct = getattr(u, "completion_tokens", None)
    if pt is None and isinstance(u, dict):
        pt = u.get("prompt_tokens", 0); ct = u.get("completion_tokens", 0)
    return int(pt or 0), int(ct or 0)

def chat(system, user, model=os.getenv("MODEL_NAME", "gpt-4o-mini")):
    # ---- MOCK MODE: no external calls, zero cost ----
    if os.getenv("MOCK", "0") == "1":
        t0 = time.perf_counter()
        m = re.search(r"\[(\d+)\]", user)
        cite = f"[{m.group(1)}]" if m else "[1]"
        msg = f"The answer is in the provided context {cite}."
        dt = time.perf_counter() - t0
        log_row(model=f"MOCK:{model}", latency_s=f"{dt:.3f}",
                prompt_tok=0, completion_tok=0, cost_usd="0.000000", cache_hit="MOCK")
        return msg

    # ---- normal path (uses cache + logs) ----
    key = key_hash(model, system, user)
    if os.getenv("CACHE", "1") == "1":
        hit = cache_get(key)
        if hit is not None:
            log_row(model=model, latency_s="0.000", prompt_tok=0, completion_tok=0, cost_usd="0.000000", cache_hit=1)
            return hit

    t0 = time.perf_counter()
    resp = _client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
    )
    dt = time.perf_counter() - t0
    msg = resp.choices[0].message.content.strip()
    in_tok, out_tok = _tokens_from_usage(getattr(resp, "usage", None))
    pr = _PRICE.get(model, {"in": 0.0, "out": 0.0})
    cost = (in_tok/1000.0)*pr["in"] + (out_tok/1000.0)*pr["out"]
    log_row(model=model, latency_s=f"{dt:.3f}", prompt_tok=in_tok, completion_tok=out_tok, cost_usd=f"{cost:.6f}", cache_hit=0)
    if os.getenv("CACHE", "1") == "1":
        cache_put(key, msg)
    return msg
