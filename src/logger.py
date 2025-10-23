import csv, os, hashlib, json

LOG_PATH = "artifacts/llm_log.csv"
CACHE_PATH = "artifacts/cache.jsonl"

def _ensure_dirs():
    os.makedirs("artifacts", exist_ok=True)

def log_row(**kw):
    _ensure_dirs()
    write_header = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sorted(kw.keys()))
        if write_header:
            w.writeheader()
        w.writerow(kw)

def cache_get(key_hash):
    if not os.path.exists(CACHE_PATH):
        return None
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("key") == key_hash:
                return obj.get("answer")
    return None

def cache_put(key_hash, answer):
    _ensure_dirs()
    with open(CACHE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key_hash, "answer": answer}) + "\n")

def key_hash(model, system, user):
    h = hashlib.sha256()
    h.update((model + "||" + system + "||" + user).encode("utf-8"))
    return h.hexdigest()
