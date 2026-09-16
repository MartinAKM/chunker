"""
Local server for chunk_viewer.html with semantic search.

    python viewer_server.py                                  # chunks_titled.jsonl + embeddings.*
    python viewer_server.py --chunks chunks.jsonl --port 8765 --no-browser

Opens http://localhost:8765 in the browser. The page loads the chunks
automatically, and semantic search runs here in Python with the same ONNX
model as cluster.py (taken from the local Hugging Face cache, nothing is
downloaded again). Only this computer can reach the server (127.0.0.1).

Endpoints used by the page:
    GET  /                 the viewer
    GET  /api/info         which chunks / embeddings / model are loaded
    GET  /api/chunks       the chunks file
    POST /api/search       {"query": "...", "k": 20} -> [{"chunk_id", "score"}]
"""
import argparse, json, os, sys, threading, time, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import embeddings_store as store

ap = argparse.ArgumentParser()
ap.add_argument("--chunks", default="chunks_titled.jsonl")
ap.add_argument("--embeddings", default="embeddings")
ap.add_argument("--viewer", default=os.path.join(HERE, "chunk_viewer.html"))
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--no-browser", action="store_true")
args = ap.parse_args()

for f in (args.chunks, args.viewer):
    if not os.path.exists(f):
        sys.exit(f"File not found: {f}")

emb = store.load(args.embeddings, required=False)
state = {"model": None, "error": None}
lock = threading.Lock()

def check_ids():
    if not emb: return 0
    ids = set()
    with open(args.chunks, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids.add(json.loads(line)["chunk_id"])
    return sum(1 for c in emb["meta"]["chunk_ids"] if c not in ids)

missing = check_ids()

def load_model():
    """Load the embedding model once, in the background, so the page opens at once."""
    try:
        t = time.time()
        store.embed_query(emb, "aquecimento")          # loads and warms up the model
        state["model"] = "ready"
        print(f"model ready ({time.time() - t:.1f}s)")
    except SystemExit as e:
        state["error"] = str(e)
    except Exception as e:
        state["error"] = f"{type(e).__name__}: {e}"
    if state["error"]:
        print("model error:", state["error"])

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send(200, open(args.viewer, "rb").read(), "text/html; charset=utf-8")
        elif self.path == "/api/info":
            info = {"chunks_file": os.path.abspath(args.chunks), "embeddings": bool(emb),
                    "model_state": "ready" if state["model"] else ("error" if state["error"] else "loading"),
                    "error": state["error"], "missing_ids": missing}
            if emb:
                m = emb["meta"]
                info.update(model=m["model"], onnx_file=m.get("onnx_file"), count=m["count"], dims=m["dims"])
            else:
                info["error"] = (f"No embeddings '{args.embeddings}.npy/.json' found. "
                                 f"Run cluster.py with the ONNX model first.")
            self.send(200, info)
        elif self.path == "/api/chunks":
            self.send(200, open(args.chunks, "rb").read(), "application/x-ndjson; charset=utf-8")
        else:
            self.send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/search":
            return self.send(404, {"error": "not found"})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            query, k = str(req.get("query", "")).strip(), max(1, min(int(req.get("k", 20)), 200))
        except Exception:
            return self.send(400, {"error": "invalid request"})
        if not emb:
            return self.send(409, {"error": "no embeddings loaded"})
        if state["error"]:
            return self.send(500, {"error": state["error"]})
        if not query:
            return self.send(400, {"error": "empty query"})
        t = time.time()
        with lock:                                       # one ONNX run at a time
            results = store.search(emb, query, k=k)
        self.send(200, {"results": [{"chunk_id": c, "score": round(s, 4)} for c, s in results],
                        "ms": int((time.time() - t) * 1000)})

threading.Thread(target=load_model, daemon=True).start() if emb else None
server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
url = f"http://localhost:{args.port}/"
print(f"chunks: {args.chunks}")
print(f"embeddings: {args.embeddings + '.npy/.json' if emb else 'NOT FOUND (semantic search disabled)'}")
if missing:
    print(f"warning: {missing} embedded chunk ids are not in {args.chunks}; re-run cluster.py")
print(f"viewer: {url}   (Ctrl+C to stop)")
if not args.no_browser:
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
try:
    server.serve_forever()
except KeyboardInterrupt:
    print("\nstopped")
