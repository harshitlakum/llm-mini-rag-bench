import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from src.retriever import TfidfRetriever

def _fake_index(p: Path):
    docs = [
        {"id": "a.txt", "text": "alpha beta\n\nalpha line two"},
        {"id": "b.txt", "text": "gamma delta alpha"},
    ]
    owners = [
        {"doc_id": "a.txt", "start": 0, "end": 11},  # "alpha beta"
        {"doc_id": "b.txt", "start": 0, "end": 18},  # "gamma delta alpha"
    ]
    chunks = [docs[0]["text"][0:11], docs[1]["text"][0:18]]
    v = TfidfVectorizer().fit(chunks)
    X = v.transform(chunks)
    obj = {"docs": docs, "owners": owners, "vec": v, "X": X}
    pickle.dump(obj, open(p, "wb"))

def test_topk_runs(tmp_path):
    p = tmp_path / "tfidf.pkl"
    _fake_index(p)
    r = TfidfRetriever(str(p))
    out = r.topk("alpha", k=2)
    assert len(out) == 2
    assert out[0]["score"] >= out[1]["score"]
    assert {"id","start","end","text","score"} <= set(out[0].keys())
