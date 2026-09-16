"""
Text embeddings with ONNX Runtime (no torch / sentence-transformers).

    pip install onnxruntime tokenizers huggingface_hub numpy

Default model: raludi/bge-m3-onnx-int8 (BGE-M3, int8 quantized, 1024 dims, ~570 MB).
Also works with onnx-community/multilingual-e5-base-ONNX and similar exports
(repo with onnx/<file>.onnx + tokenizer.json).
The model files are downloaded once to the Hugging Face cache.

    from embedder import Embedder
    emb = Embedder()                                   # or Embedder(onnx_file="onnx/model_quantized.onnx")
    P = emb.encode(["texto do chunk"], kind="passage") # documents
    q = emb.encode(["minha pergunta"], kind="query")   # questions
    # both are L2-normalised: similarity = P @ q[0]

Model-specific details are handled automatically:
- pooling: read from 1_Pooling/config.json when the repo has it; otherwise
  CLS for BGE models and mean for E5 (how each model was trained)
- prefixes: E5 needs "passage: " / "query: "; BGE-M3 needs none

Command line check:
    python embedder.py "como liberar pedido"     # prints the vector size and a few values
"""
import os, sys, time
import numpy as np

DEFAULT_MODEL = "raludi/bge-m3-onnx-int8"
DEFAULT_FILE = "onnx/model.onnx"

class Embedder:
    def __init__(self, model=DEFAULT_MODEL, onnx_file=DEFAULT_FILE, max_length=None, threads=None,
                 local_dir=None):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        name = model.lower()
        # BGE-M3 reads up to 8192 tokens; 2048 covers the largest chunks (~6000 chars)
        # without the memory cost of the full length. E5 is limited to 512.
        if max_length is None:
            max_length = 2048 if "bge-m3" in name else 512
        self.model, self.onnx_file, self.max_length = model, onnx_file, max_length
        model_path, tok_path = self._files(model, onnx_file, local_dir)
        self.pooling = self._pooling(model, local_dir) or ("cls" if "bge" in name else "mean")

        self.tok = Tokenizer.from_file(tok_path)
        self.tok.enable_truncation(max_length=max_length)
        pad_id = self.tok.token_to_id("<pad>")
        self.tok.enable_padding(pad_id=pad_id if pad_id is not None else 1,
                                pad_token="<pad>" if pad_id is not None else "[PAD]")

        so = ort.SessionOptions()
        if threads: so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.session.get_inputs()}
        outs = [o.name for o in self.session.get_outputs()]
        self.output = "last_hidden_state" if "last_hidden_state" in outs else outs[0]
        e5 = "e5" in name
        self.prefix = {"passage": "passage: " if e5 else "", "query": "query: " if e5 else ""}

    @staticmethod
    def _pooling(model, local_dir):
        """Pooling declared by the sentence-transformers config, if the repo has one."""
        import json
        path = None
        base = local_dir or (model if os.path.isdir(model) else None)
        if base:
            p = os.path.join(base, "1_Pooling", "config.json")
            path = p if os.path.exists(p) else None
        else:
            from huggingface_hub import hf_hub_download
            for local_only in (True, False):
                try:
                    path = hf_hub_download(model, "1_Pooling/config.json", local_files_only=local_only)
                    break
                except Exception:
                    pass
        if not path:
            return None
        cfg = json.load(open(path, encoding="utf-8"))
        if cfg.get("pooling_mode_cls_token"): return "cls"
        if cfg.get("pooling_mode_mean_tokens"): return "mean"
        return None

    @staticmethod
    def _files(model, onnx_file, local_dir):
        """Local folder (same layout as the repo) or download from Hugging Face."""
        if local_dir:
            return os.path.join(local_dir, onnx_file), os.path.join(local_dir, "tokenizer.json")
        if os.path.isdir(model):
            return os.path.join(model, onnx_file), os.path.join(model, "tokenizer.json")
        from huggingface_hub import hf_hub_download
        # use the local Hugging Face cache first (no network), download only if missing
        try:
            m = hf_hub_download(model, onnx_file, local_files_only=True)
            tok = hf_hub_download(model, "tokenizer.json", local_files_only=True)
            try:
                hf_hub_download(model, onnx_file + "_data", local_files_only=True)
            except Exception:
                pass
            return m, tok
        except Exception:
            pass
        try:
            m = hf_hub_download(model, onnx_file)
        except Exception as e:
            try:
                from huggingface_hub import list_repo_files
                files = [f for f in list_repo_files(model) if f.endswith(".onnx")]
                hint = "Available .onnx files: " + ", ".join(files)
            except Exception:
                hint = ""
            sys.exit(f"Could not get '{onnx_file}' from {model}: {e}\n{hint}")
        # some exports keep weights in a separate file next to the .onnx
        try:
            hf_hub_download(model, onnx_file + "_data")
        except Exception:
            pass
        return m, hf_hub_download(model, "tokenizer.json")

    def encode(self, texts, kind="passage", batch_size=8, progress=False):
        if isinstance(texts, str): texts = [texts]
        texts = [self.prefix[kind] + t for t in texts]
        # sort by length so each batch pads to a similar size (much faster)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = np.zeros((len(texts), 0), dtype=np.float32)
        result, t0 = [None] * len(texts), time.time()
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            enc = self.tok.encode_batch([texts[i] for i in idx])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            hidden = self.session.run([self.output], {k: v for k, v in feed.items() if k in self.inputs})[0]
            if hidden.ndim == 3 and self.pooling == "cls":        # first token (<s>)
                vec = hidden[:, 0]
            elif hidden.ndim == 3:                                 # mean of the real tokens
                m = mask[..., None].astype(np.float32)
                vec = (hidden * m).sum(1) / np.maximum(m.sum(1), 1e-9)
            else:                                                  # model already pooled
                vec = hidden
            vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
            for i, v in zip(idx, vec): result[i] = v.astype(np.float32)
            if progress:
                done = min(b + batch_size, len(texts))
                el = time.time() - t0
                eta = el / done * (len(texts) - done)
                print(f"\r  embedded {done}/{len(texts)}  {el:.0f}s elapsed, ~{eta:.0f}s left",
                      end="", flush=True)
        if progress: print()
        return np.vstack(result) if result else out

if __name__ == "__main__":
    e = Embedder()
    v = e.encode(" ".join(sys.argv[1:]) or "teste", kind="query")
    print(f"model {e.model} | pooling {e.pooling} | max tokens {e.max_length} | vector {v.shape}")
    print(v[0][:5])
