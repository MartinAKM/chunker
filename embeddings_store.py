"""
Saved chunk embeddings, reusable by any script.

cluster.py writes them (embeddings.npy + embeddings.json) and only re-embeds
chunks that are new or changed. To use them elsewhere:

    import embeddings_store as store

    emb = store.load()                       # vectors + metadata
    emb["vectors"]                           # numpy array (chunks x dims), normalised
    emb["meta"]["chunk_ids"]                 # chunk id of each row
    store.vector_of(emb, "0-1")              # the vector of one chunk

    # search with a new text (uses the same model the vectors were made with)
    for chunk_id, score in store.search(emb, "como liberar pedido para faturamento", k=5):
        print(chunk_id, score)

    # chunks most similar to an existing chunk
    store.similar_to(emb, "0-1", k=5)

Command line:
    python embeddings_store.py info
    python embeddings_store.py search "como liberar pedido para faturamento"
    python embeddings_store.py similar 0-1
"""
import hashlib, json, os, sys
import numpy as np

def text_hash(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

def save(name, vectors, meta):
    vectors = np.asarray(vectors, dtype=np.float32)
    assert len(vectors) == len(meta["chunk_ids"]), "one vector per chunk id"
    meta = dict(meta, dims=int(vectors.shape[1]), count=int(len(vectors)))
    np.save(name + ".tmp.npy", vectors)                 # write both, then swap: never half-saved
    with open(name + ".tmp.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    os.replace(name + ".tmp.npy", name + ".npy")
    os.replace(name + ".tmp.json", name + ".json")

def load(name="embeddings", required=True):
    if not (os.path.exists(name + ".npy") and os.path.exists(name + ".json")):
        if required:
            sys.exit(f"No saved embeddings '{name}'. Run cluster.py with an embedding model first.")
        return None
    vectors = np.load(name + ".npy")
    meta = json.load(open(name + ".json", encoding="utf-8"))
    if len(vectors) != len(meta["chunk_ids"]):
        sys.exit(f"'{name}' is inconsistent ({len(vectors)} vectors, {len(meta['chunk_ids'])} ids).")
    return {"vectors": vectors, "meta": meta,
            "index": {c: i for i, c in enumerate(meta["chunk_ids"])}}

def vector_of(emb, chunk_id):
    return emb["vectors"][emb["index"][chunk_id]]

_models = {}
def embed_query(emb, text):
    """Embed a new text with the SAME model and prefix used for the saved vectors."""
    m = emb["meta"]
    if m.get("backend") != "onnx":
        raise ValueError(f"these embeddings were made with backend '{m.get('backend')}'; "
                         f"run cluster.py again to rebuild them with ONNX")
    from embedder import Embedder
    key = (m["model"], m["onnx_file"])
    if key not in _models:
        _models[key] = Embedder(m["model"], m["onnx_file"])
    return _models[key].encode([text], kind="query")[0]

def top(emb, vector, k=5, exclude=None):
    sims = emb["vectors"] @ vector
    out = []
    for i in np.argsort(-sims):
        cid = emb["meta"]["chunk_ids"][i]
        if cid == exclude: continue
        out.append((cid, float(sims[i])))
        if len(out) == k: break
    return out

def search(emb, text, k=5):
    return top(emb, embed_query(emb, text), k)

def similar_to(emb, chunk_id, k=5):
    return top(emb, vector_of(emb, chunk_id), k, exclude=chunk_id)

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in {"info", "search", "similar"}:
        sys.exit(__doc__)
    emb = load()
    if sys.argv[1] == "info":
        m = emb["meta"]
        print(f"model: {m['model']} | chunks: {m['count']} | dims: {m['dims']} | "
              f"text: {m['text_mode']} | from: {m['chunks_file']}")
    elif sys.argv[1] == "search":
        for cid, s in search(emb, " ".join(sys.argv[2:]), k=10): print(f"{s:.3f}  {cid}")
    else:
        for cid, s in similar_to(emb, sys.argv[2], k=10): print(f"{s:.3f}  {cid}")
