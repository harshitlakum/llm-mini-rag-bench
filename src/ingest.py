import os, glob, pickle, re
from sklearn.feature_extraction.text import TfidfVectorizer

def load_docs(folder):
    files = sum([glob.glob(os.path.join(folder, f"*.{ext}")) for ext in ("txt","md")], [])
    docs = []
    for fp in sorted(files):
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        docs.append({"id": os.path.basename(fp), "text": txt})
    return docs

def chunk_doc(doc_text, min_len=300, max_len=900):
    # paragraph-ish chunks with char offsets
    paras = [p.strip() for p in re.split(r"\n\s*\n", doc_text) if p.strip()]
    chunks, pos = [], 0
    for p in paras:
        start = doc_text.find(p, pos)
        end = start + len(p)
        pos = end
        # split long paragraphs
        for i in range(0, len(p), max_len):
            seg = p[i:i+max_len]
            if len(seg) < min_len and i > 0: break
            seg_start = start + i
            seg_end = seg_start + len(seg)
            chunks.append({"start": seg_start, "end": seg_end, "text": seg})
    # fallback: single chunk if very short
    if not chunks and doc_text.strip():
        chunks.append({"start": 0, "end": len(doc_text), "text": doc_text})
    return chunks

if __name__ == "__main__":
    docs = load_docs("data/docs")
    if not docs:
        raise SystemExit("No docs found in data/docs")
    all_chunks, owners = [], []
    for d in docs:
        chs = chunk_doc(d["text"])
        for c in chs:
            all_chunks.append(c["text"])
            owners.append({"doc_id": d["id"], "start": c["start"], "end": c["end"]})
    vec = TfidfVectorizer(ngram_range=(1,2), max_df=0.9, min_df=1)
    X = vec.fit_transform(all_chunks)
    os.makedirs("artifacts", exist_ok=True)
    pickle.dump({"docs": docs, "owners": owners, "vec": vec, "X": X}, open("artifacts/tfidf.pkl","wb"))
    print(f"Indexed {len(docs)} docs → {len(all_chunks)} chunks → artifacts/tfidf.pkl")
