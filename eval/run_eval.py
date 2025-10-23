import json, re
from src.rag import answer, get_ret
from src.llm import chat
from src.prompts import SYSTEM

def f1_keywords(pred, keywords):
    pred_tokens = set(re.findall(r"\w+", pred.lower()))
    gold = set(k.lower() for k in keywords)
    if not gold:
        return 0.0
    tp = len(pred_tokens & gold)
    prec = tp / max(1, len(pred_tokens))
    rec  = tp / len(gold)
    return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)

def run_no_retrieval(q):
    user = f"Question: {q}\n\nAnswer briefly."
    return chat(SYSTEM, user)

if __name__ == "__main__":
    data = [json.loads(x) for x in open("eval/eval.jsonl","r")]
    ret = get_ret()
    hits, f1_rag, f1_base = [], [], []

    for ex in data:
        q = ex["q"]
        kw = ex.get("keywords", [])
        rag_ans = answer(q)["answer"]
        base_ans = run_no_retrieval(q)

        top = ret.topk(q, k=5)
        text = " ".join([t["text"].lower() for t in top])
        hit = float(all(k.lower() in text for k in kw)) if kw else 0.0

        hits.append(hit)
        f1_rag.append(f1_keywords(rag_ans, kw))
        f1_base.append(f1_keywords(base_ans, kw))

    print(f"hit@5={sum(hits)/len(hits):.3f} | F1_rag={sum(f1_rag)/len(f1_rag):.3f} | F1_base={sum(f1_base)/len(f1_base):.3f}")
