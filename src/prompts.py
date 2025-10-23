SYSTEM = """Answer ONLY from the provided context. If missing, say:
"I don't have enough information." Cite snippets with [1], [2], ..."""

def build_user_prompt(question, chunks, max_chars=3000):
    if not chunks:
        return f"Context:\n\nQuestion: {question}\n\nAnswer:"
    budget = max(1, max_chars // max(1, len(chunks)))
    ctx_lines = []
    for i, c in enumerate(chunks, start=1):
        ctx = c["text"][:budget]
        ctx_lines.append(f"[{i}] (source: {c['id']}:{c['start']}-{c['end']})\n{ctx}")
    ctx_blob = "\n\n---\n\n".join(ctx_lines)
    return f"Context:\n{ctx_blob}\n\nQuestion: {question}\n\nAnswer (use [#] citations):"
