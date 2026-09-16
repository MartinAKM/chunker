"""
Cluster the chunks and check how well they are separated and linked.

    pip install scikit-learn numpy
    pip install onnxruntime tokenizers huggingface_hub umap-learn   # recommended, optional

    python cluster.py chunks_titled.jsonl                     # embeddings (downloads the model once)
    python cluster.py chunks_titled.jsonl --tfidf             # no model, word statistics only
    python cluster.py chunks_titled.jsonl --onnx-file onnx/model_quantized.onnx   # faster

Embeddings are saved next to the script as embeddings.npy + embeddings.json
(--embeddings to change the name) and reused on the next run: only new or
changed chunks are embedded again. Use embeddings_store.py to read them.

Outputs (folder "cluster_out"):
    cluster_map.html      interactive map: every dot is a chunk, colours are clusters,
                          click a dot to see its text and its nearest chunks
    clusters.csv          chunk -> cluster, 2D position, title
    cluster_summary.csv   one row per cluster: size, keywords, main modules, purity
    chunk_links.csv       each chunk's most similar chunks (the "link" candidates)
    cluster_links.csv     how strongly clusters are connected to each other
    quality_report.txt    checks on the chunking (see the end of this file)
"""
import argparse, collections, csv, html, json, math, os, re, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("chunks", nargs="?", default="chunks_titled.jsonl")
ap.add_argument("--out", default="cluster_out")
ap.add_argument("--tfidf", action="store_true", help="use TF-IDF instead of an embedding model")
ap.add_argument("--model", default="onnx-community/multilingual-e5-base-ONNX",
                help="Hugging Face ONNX repo, or a local folder with the same layout")
ap.add_argument("--onnx-file", default="onnx/model.onnx",
                help="which .onnx inside the repo (onnx/model_quantized.onnx is smaller and faster)")
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--min-cluster", type=int, default=15, help="smallest group HDBSCAN may form")
ap.add_argument("--k", type=int, default=5, help="neighbours kept per chunk")
ap.add_argument("--with-header", action="store_true",
                help="include the chunk header (document/program) in the vectors. Off by default: "
                     "the header makes chunks of one document look alike, which hides chunking problems")
ap.add_argument("--embeddings", default="embeddings",
                help="where embeddings are saved and reused: <name>.npy + <name>.json. Only chunks "
                     "that are new or changed since the last run are embedded again")
ap.add_argument("--link-min", type=float, default=0.55, help="min similarity for a link")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

# ------------------------------------------------------------------ data
rows = [json.loads(l) for l in open(args.chunks, encoding="utf-8")]
MOD = re.compile(r"^[A-Z]+")
CAP = re.compile(r"\[Imagem:[^\]]*\]")
CODE = re.compile(r"\b[A-Z]{3,5}\d{1,4}\b")

def module_of(r):
    mods = {MOD.match(p).group(0) for p in r["main_programs"].split()}
    return mods.pop() if len(mods) == 1 else ""

def body(r):
    """Text used for clustering: the chunk body without image refs
    (and, only with --with-header, the document/program header)."""
    text = CAP.sub(" ", r["text"])
    if args.with_header:
        text = r["embed_text"].split("\n", 1)[0] + "\n" + text
    return text

texts = [body(r) for r in rows]
text_mode = "with-header" if args.with_header else "body"
N = len(rows)
log(f"{N} chunks loaded")

# ------------------------------------------------------------------ vectors
STOP = set("""a à ao aos as às até com como da das de do dos e é em entre essa esse esta este foi
há isso já mais mas na nas não no nos o os ou para pela pelo por que se sem ser seu sua são também
tem um uma el la los las del con y lo su es campo tecle clique botão lista valores tela programa
posicionar informar localizar desejado desejada efetuar selecionar abrir aba opção registro salvar
pedido serviço solicitação imagem cliente tipo tópico pdf docx documentacao""".split())

def tfidf_vectors(txts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize
    v = TfidfVectorizer(min_df=3, max_df=0.4, sublinear_tf=True, stop_words=list(STOP),
                        token_pattern=r"(?u)\b[^\W\d_]{3,}\b", ngram_range=(1, 2), max_features=60000)
    X = v.fit_transform(txts)
    return normalize(TruncatedSVD(256, random_state=0).fit_transform(X))

use_tfidf = args.tfidf
if not use_tfidf:
    try:
        import onnxruntime, tokenizers, huggingface_hub  # noqa: F401
    except ImportError:
        print("onnxruntime/tokenizers not installed -> using TF-IDF. "
              "For better clusters: pip install onnxruntime tokenizers huggingface_hub")
        use_tfidf = True

import embeddings_store as store
if use_tfidf:
    log("vectors: TF-IDF + SVD (not saved: they can't embed new text later)")
    E = tfidf_vectors(texts)
    method = "TF-IDF"
else:
    old = store.load(args.embeddings, required=False)
    reuse = {}
    if old and old["meta"]["model"] == args.model and old["meta"].get("onnx_file") == args.onnx_file \
            and old["meta"]["text_mode"] == text_mode:
        reuse = {h: old["vectors"][i] for i, h in enumerate(old["meta"]["hashes"])}
    hashes = [store.text_hash(t) for t in texts]
    todo = [i for i, h in enumerate(hashes) if h not in reuse]
    log(f"vectors: {args.model} | reused {len(texts) - len(todo)} from {args.embeddings}, "
        f"embedding {len(todo)} new/changed chunks")
    fresh = {}
    e5 = "e5" in args.model.lower()
    prefixes = {"passage": "passage: " if e5 else "", "query": "query: " if e5 else ""}
    if todo:                                   # the model is only loaded when needed
        from embedder import Embedder
        embedder = Embedder(args.model, args.onnx_file)
        prefixes = embedder.prefix
        V = embedder.encode([texts[i] for i in todo], kind="passage",
                            batch_size=args.batch, progress=True)
        fresh = {hashes[i]: v for i, v in zip(todo, V)}
    E = np.asarray([reuse.get(h, fresh.get(h)) for h in hashes], dtype=np.float32)
    method = args.model
    store.save(args.embeddings, E, {
        "model": args.model, "backend": "onnx", "onnx_file": args.onnx_file,
        "passage_prefix": prefixes["passage"], "query_prefix": prefixes["query"],
        "normalized": True,
        "text_mode": text_mode, "chunks_file": os.path.abspath(args.chunks),
        "chunk_ids": [r["chunk_id"] for r in rows], "hashes": hashes,
    })
    log(f"embeddings saved to {args.embeddings}.npy / .json")

# ------------------------------------------------------------------ reduce + cluster
try:
    import umap
    log("reducing with UMAP")
    R = umap.UMAP(n_components=15, n_neighbors=15, min_dist=0.0, metric="cosine",
                  random_state=0).fit_transform(E)
    XY = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine",
                   random_state=0).fit_transform(E)
except ImportError:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    log("umap-learn not installed -> PCA + t-SNE (slower). pip install umap-learn")
    XY = TSNE(2, init="pca", perplexity=30, random_state=0, max_iter=750).fit_transform(
        PCA(30, random_state=0).fit_transform(E))
    # density clustering does not work well in 30 dims; the t-SNE map keeps the
    # neighbourhoods, so cluster there (fine for exploration)
    R = XY

from sklearn.cluster import HDBSCAN
log("clustering with HDBSCAN")
labels = HDBSCAN(min_cluster_size=args.min_cluster, min_samples=5).fit_predict(R)
n_clusters = labels.max() + 1
log(f"{n_clusters} clusters, {np.sum(labels < 0)} chunks without a cluster (noise)")

# ------------------------------------------------------------------ cluster names (class-based TF-IDF)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(min_df=3, max_df=0.3, stop_words=list(STOP),
                     token_pattern=r"(?u)\b[^\W\d_]{3,}\b", ngram_range=(1, 2))
plain = [CODE.sub(" ", CAP.sub(" ", r["text"])) for r in rows]
C = cv.fit_transform(plain)
vocab = np.array(cv.get_feature_names_out())
ctf = np.zeros((n_clusters, C.shape[1]))
for c in range(n_clusters):
    ctf[c] = np.asarray(C[labels == c].sum(0)).ravel()
tf = ctf / np.maximum(ctf.sum(1, keepdims=True), 1)
idf = np.log(1 + ctf.sum(1).mean() / np.maximum(ctf.sum(0), 1))
score = tf * idf
keywords = {}
for c in range(n_clusters):
    kw = []
    for i in score[c].argsort()[::-1]:
        w = vocab[i]
        if any(w in k or k in w for k in kw): continue
        kw.append(w)
        if len(kw) == 6: break
    keywords[c] = kw

# ------------------------------------------------------------------ neighbours / links
log("finding nearest neighbours")
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=args.k + 1, metric="cosine").fit(E)
dist, idx = nn.kneighbors(E)
sims = 1 - dist

def relation(a, b):
    ra, rb = rows[a], rows[b]
    if ra["doc_id"] != rb["doc_id"]: return "other_document"
    if ra.get("topic_id") and ra.get("topic_id") == rb.get("topic_id"): return "same_topic"
    if ra["record"] == rb["record"]: return "same_record"
    return "same_document"

links = []
for a in range(N):
    for s, b in zip(sims[a][1:], idx[a][1:]):
        if s >= args.link_min:
            links.append((a, int(b), float(s), relation(a, int(b))))

prog_sets = [set(r["mentioned_programs"].split()) | set(r["main_programs"].split()) for r in rows]
with open(os.path.join(args.out, "chunk_links.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["from_chunk", "to_chunk", "similarity", "relation", "shared_programs",
                "from_cluster", "to_cluster", "from_title", "to_title"])
    for a, b, s, rel in links:
        w.writerow([rows[a]["chunk_id"], rows[b]["chunk_id"], f"{s:.3f}", rel,
                    " ".join(sorted(prog_sets[a] & prog_sets[b])),
                    labels[a], labels[b], rows[a].get("title", ""), rows[b].get("title", "")])

cl_links = collections.Counter()
for a, b, s, rel in links:
    ca, cb = labels[a], labels[b]
    if ca >= 0 and cb >= 0 and ca != cb:
        cl_links[tuple(sorted((int(ca), int(cb))))] += 1

# ------------------------------------------------------------------ summaries
modules = [module_of(r) for r in rows]
summary = []
for c in range(n_clusters):
    members = np.where(labels == c)[0]
    mods = collections.Counter(modules[i] for i in members if modules[i])
    types = collections.Counter(rows[i]["record_type"] for i in members)
    docs = collections.Counter(rows[i]["doc_id"] for i in members)
    labelled = sum(mods.values())
    top_mod, top_n = mods.most_common(1)[0] if mods else ("", 0)
    centre = E[members].mean(0); centre /= np.linalg.norm(centre) + 1e-9
    cohesion = float((E[members] @ centre).mean())
    example = rows[members[np.argmax(E[members] @ centre)]].get("title", "")
    summary.append({
        "cluster": c, "size": len(members), "keywords": ", ".join(keywords[c]),
        "top_module": top_mod, "module_purity": round(top_n / labelled, 2) if labelled else "",
        "modules": " ".join(f"{m}:{n}" for m, n in mods.most_common(4)),
        "record_types": " ".join(f"{t}:{n}" for t, n in types.most_common()),
        "documents": len(docs), "largest_doc_share": round(docs.most_common(1)[0][1] / len(members), 2),
        "cohesion": round(cohesion, 3), "example": example,
    })
with open(os.path.join(args.out, "cluster_summary.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)
with open(os.path.join(args.out, "cluster_links.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["cluster_a", "cluster_b", "links", "keywords_a", "keywords_b"])
    for (a, b), n in cl_links.most_common():
        w.writerow([a, b, n, ", ".join(keywords[a][:3]), ", ".join(keywords[b][:3])])
with open(os.path.join(args.out, "clusters.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["chunk_id", "cluster", "x", "y", "doc_name", "record_type", "module", "title"])
    for i, r in enumerate(rows):
        w.writerow([r["chunk_id"], labels[i], f"{XY[i,0]:.3f}", f"{XY[i,1]:.3f}", r["doc_name"],
                    r["record_type"], modules[i], r.get("title", "")])

# ------------------------------------------------------------------ quality checks on the chunking
from sklearn.metrics import normalized_mutual_info_score
lines = []
def say(s=""): lines.append(s)

say(f"Method: {method} | chunks: {N} | clusters: {n_clusters} | noise: {int(np.sum(labels<0))} "
    f"({np.mean(labels<0):.0%})")
lab_idx = [i for i in range(N) if modules[i] and labels[i] >= 0]
if lab_idx:
    say(f"Agreement with program modules (NMI, 0-1): "
        f"{normalized_mutual_info_score([modules[i] for i in lab_idx], [labels[i] for i in lab_idx]):.3f}")
say()

# 1. neighbours: where do a chunk's most similar chunks come from?
rel = collections.Counter(l[3] for l in links)
say("1. Nearest-neighbour links by relation (a healthy mix has many 'other_document' links;")
say("   almost only 'same_*' means chunks are mostly similar to their own document):")
for k, v in rel.most_common(): say(f"   {k:16s} {v}")
say()

# 2. near-duplicates: probably repeated text that the cleaning missed, or copied docs
dups = [(a, b, s) for a, b, s, _ in links if s >= 0.97 and a < b]
say(f"2. Near-duplicate pairs (similarity >= 0.97): {len(dups)}")
for a, b, s in sorted(dups, key=lambda x: -x[2])[:25]:
    say(f"   {s:.3f}  {rows[a]['chunk_id']:>8} {rows[a]['doc_name'][:40]:40s} | "
        f"{rows[b]['chunk_id']:>8} {rows[b]['doc_name'][:40]}")
say()

# 3. topics split across clusters: a numbered topic whose chunks landed in different clusters
by_topic = collections.defaultdict(list)
for i, r in enumerate(rows):
    if r.get("topic_number"): by_topic[r["topic_id"]].append(i)
split = [(t, ids) for t, ids in by_topic.items()
         if len(ids) > 1 and len({labels[i] for i in ids if labels[i] >= 0}) > 1]
say(f"3. Numbered topics with more than one chunk: {sum(len(v)>1 for v in by_topic.values())}; "
    f"spread over different clusters: {len(split)}")
for t, ids in split[:15]:
    say(f"   {rows[ids[0]]['doc_name'][:45]:45s} topic {rows[ids[0]]['topic_number']}: "
        f"clusters {[int(labels[i]) for i in ids]}")
say()

# 4. consecutive chunks of the same record: very low similarity -> maybe two subjects mixed in one record
#    very high similarity -> maybe a needless cut
low, high = [], []
for i in range(N - 1):
    a, b = rows[i], rows[i + 1]
    if a["doc_id"] == b["doc_id"] and a["record"] == b["record"]:
        s = float(E[i] @ E[i + 1])
        if s < 0.25: low.append((s, i))
        if s > 0.92: high.append((s, i))
say(f"4. Consecutive chunks in the same record with very LOW similarity (<0.25): {len(low)}")
say("   -> the record may contain two different subjects that should be separate")
for s, i in sorted(low)[:15]:
    say(f"   {s:.2f}  {rows[i]['chunk_id']} -> {rows[i+1]['chunk_id']}  {rows[i]['doc_name'][:50]}")
say(f"   Consecutive chunks with very HIGH similarity (>0.92): {len(high)}")
say("   -> probably one subject cut in two; candidates for joining")
for s, i in sorted(high, reverse=True)[:15]:
    say(f"   {s:.2f}  {rows[i]['chunk_id']} -> {rows[i+1]['chunk_id']}  {rows[i]['doc_name'][:50]}")
say()

# 5. clusters dominated by one document or one record type (form, not subject)
say("5. Clusters that group by FORM rather than subject:")
for srow in summary:
    rt = srow["record_types"].split()[0]
    t, n = rt.split(":")
    if srow["largest_doc_share"] >= 0.8:
        say(f"   cluster {srow['cluster']:3d} ({srow['size']} chunks): {srow['largest_doc_share']:.0%} "
            f"from ONE document  [{srow['keywords']}]")
    elif t == "email" and int(n) / srow["size"] >= 0.8:
        say(f"   cluster {srow['cluster']:3d} ({srow['size']} chunks): mostly e-mails  [{srow['keywords']}]")
say()

# 6. strongest connections between clusters
say("6. Strongest links between clusters (candidate module-to-module paths):")
for (a, b), n in cl_links.most_common(20):
    say(f"   {n:4d}  [{a}] {', '.join(keywords[a][:3])}  <->  [{b}] {', '.join(keywords[b][:3])}")

report = "\n".join(lines)
open(os.path.join(args.out, "quality_report.txt"), "w", encoding="utf-8").write(report)
print(report)

# ------------------------------------------------------------------ interactive map
log("writing cluster_map.html")
pts = []
nbrs = {}
for a, b, s, rel_ in links:
    nbrs.setdefault(a, []).append([b, round(s, 3), rel_])
for i, r in enumerate(rows):
    pts.append([round(float(XY[i, 0]), 2), round(float(XY[i, 1]), 2), int(labels[i]), r["chunk_id"],
                r.get("title", "")[:140], r["doc_name"][:90], r["record_type"], modules[i],
                CAP.sub("", r["text"])[:600], nbrs.get(i, [])])
data = {"points": pts, "clusters": summary, "method": method,
        "clusterLinks": [[a, b, n] for (a, b), n in cl_links.most_common(300)]}
tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cluster_map_template.html"),
           encoding="utf-8").read()
with open(os.path.join(args.out, "cluster_map.html"), "w", encoding="utf-8") as f:
    f.write(tpl.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")))
log(f"done -> {args.out}/")
