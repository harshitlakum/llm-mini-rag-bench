import pickle, numpy as np
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import TruncatedSVD

class TfidfRetriever:
    def __init__(self, path="artifacts/tfidf.pkl"):
        obj = pickle.load(open(path, "rb"))
        self.docs = {d["id"]: d for d in obj["docs"]}
        self.owners, self.vec, self.X = obj["owners"], obj["vec"], obj["X"]

    def _build(self, i, sim_i):
        own = self.owners[i]
        doc = self.docs[own["doc_id"]]
        text = doc["text"][own["start"]:own["end"]]
        return {
            "id": own["doc_id"],
            "start": own["start"],
            "end": own["end"],
            "text": text,
            "score": float(sim_i),
        }

    def topk(self, query, k=4, rerank=False, dims=128):
        qv = self.vec.transform([query])
        sims = linear_kernel(qv, self.X).ravel()
        idx = np.argsort(-sims)[: max(k * 3, k)]
        if rerank and len(idx) > 1:
            n_comp = min(dims, max(1, min(self.X.shape[1]-1, len(idx)-1)))
            Xc = self.X[idx]
            svd = TruncatedSVD(n_components=n_comp, random_state=0)
            Xr = svd.fit_transform(Xc)
            qr = svd.transform(qv)
            rsims = (qr @ Xr.T).ravel()
            order = np.argsort(-rsims)[:k]
            idx = idx[order]
        else:
            idx = idx[:k]
        return [self._build(int(i), sims[int(i)]) for i in idx]
