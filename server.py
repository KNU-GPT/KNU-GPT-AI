from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import time
import uvicorn
#ㅇ
TREE_FILE = 'index_tree_db.json'
FAISS_INDEX_FILE = 'faiss.index'
MODEL_NAME = 'BM-K/KoSimCSE-roberta-multitask'

app = FastAPI()

# CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 트리 로드
with open(TREE_FILE, 'r', encoding='utf-8') as f:
    TREE_DB = json.load(f)

model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(FAISS_INDEX_FILE)

def collect_records(node):
    """선택된 노드 아래 모든 레코드를 리스트로 수집"""
    if isinstance(node, list):
        return node
    elif isinstance(node, dict):
        recs = []
        for v in node.values():
            recs.extend(collect_records(v))
        return recs
    return []

@app.get("/tree")
def get_tree():
    """전체 트리 구조 반환"""
    return TREE_DB

@app.post("/search")
async def search(request: Request):
    data = await request.json()
    query = data.get("query")
    path = data.get("path", [])
    top_k = int(data.get("top_k", 5))
    threshold = float(data.get("threshold", 0.6))

    node = TREE_DB
    for key in path:
        if key not in node:
            return JSONResponse({"error": f"'{key}' 경로를 찾을 수 없음"}, status_code=400)
        node = node[key]

    records = collect_records(node)
    if not records:
        return {"results": [], "time": 0, "count": 0}

    idxs = [r['vector_idx'] for r in records]
    vectors = np.array([index.reconstruct(i) for i in idxs]).astype('float32')
    faiss.normalize_L2(vectors)

    q_vec = model.encode(query, normalize_embeddings=True).astype('float32').reshape(1, -1)
    faiss.normalize_L2(q_vec)

    sub_index = faiss.IndexFlatIP(vectors.shape[1])
    sub_index.add(vectors)

    start = time.time()
    D, I = sub_index.search(q_vec, len(records))
    end = time.time()

    results = [(float(D[0][i]), records[I[0][i]]) for i in range(len(records)) if D[0][i] >= threshold]
    results = sorted(results, key=lambda x: x[0], reverse=True)[:top_k]

    return {
        "results": [
            {"score": s, "title": r["title"], "paragraph": r["paragraph"][:300]}
            for s, r in results
        ],
        "time": round(end - start, 4),
        "count": len(records)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
