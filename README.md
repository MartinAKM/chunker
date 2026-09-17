python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python chunker.py export.json
python titler.py chunks.jsonl
python cluster.py chunks_titled.jsonl

python association.py chunks_titled.jsonl --messages --terms terms.txt

python viewer_server.py


---------

python cluster.py chunks_titled.jsonl --embeddings emb_bge --out cluster_bge
python viewer_server.py --embeddings emb_bge

python cluster.py chunks_titled.jsonl --model onnx-community/multilingual-e5-base-ONNX --embeddings emb_e5 --out cluster_e5
python viewer_server.py --embeddings emb_e5 --port 8766