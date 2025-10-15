"""
=================== 입력 예시 ===================
POST /search
Content-Type: application/json

{
    "query": "성적과 상관없이 받을수 있는 장학금이 뭐가 있어?",
    "path": ["IT대학","전자공학부", "장학금"],
    "top_k": 5,
    "threshold": 0.5
}

=================== 출력 예시 ===================
{
    "results": [
        {
            "score": 0.9,
            "title": "[근로] 2025. 2학기 국가근로장학생 (전공실험실, 학부사무실 근무) 신청 [~8.18/월 13시]",
            "paragraph": "2025. 2학기 국가근로장학생 선발 안내\n\n근로기간: 9.1(월) ~ 12.19(금)\n- 한국장학재단 2025. 2학기 국가근로장학금 1차 신청자\n- 2025. 2학기 재학생\n..."
        },
        {
            "score": 0.8,
            "title": "(금)☆ [의치약학·생활비 500만원] 2025학년도 S-L생계지원 장학금 추천: 8.12.(화) 9:00까지",
            "paragraph": "추천 희망자는 8.12(화) 오전 9시까지 학부사무실(T1-409-1, ☎950-6607)로 서류 제출\n..."
        }
    ],
    "time": 0.0123,
    "count": 10
}
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import faiss
import numpy as np
import time
from sentence_transformers import SentenceTransformer
import uvicorn
import json

app = FastAPI()

# ================== 설정 ==================
INDEX_PATH = "faiss.index"             # ✅ FAISS 인덱스 파일 경로
TREE_DB_PATH = "index_tree_db.json"    # ✅ 트리형 DB JSON 경로

# ================== 모델 & 인덱스 로드 ==================
print("🚀 모델 로드 중...")
model = SentenceTransformer("BM-K/KoSimCSE-roberta-multitask")
print("🚀 FAISS 인덱스 로드 중...")
index = faiss.read_index(INDEX_PATH)

with open(TREE_DB_PATH, "r", encoding="utf-8") as f:
    TREE_DB = json.load(f)

print("✅ 검색 서버 준비 완료!")

# ================== 레코드 수집 함수 ==================
def collect_records(node):
    records = []
    if isinstance(node, dict):
        for v in node.values():
            records.extend(collect_records(v))
    elif isinstance(node, list):
        records.extend(node)
    return records

# ================== 검색 API ==================
@app.post("/search")
async def search(request: Request):
    data = await request.json()
    query = data.get("query")
    path = data.get("path", [])
    top_k = int(data.get("top_k", 5))
    threshold = float(data.get("threshold", 0.6))

    # 1️⃣ 경로 탐색
    node = TREE_DB
    for key in path:
        if key not in node:
            return JSONResponse({"error": f"'{key}' 경로를 찾을 수 없음"}, status_code=400)
        node = node[key]

    # 2️⃣ 레코드 수집
    records = collect_records(node)
    if not records:
        return {"results": [], "time": 0, "count": 0}

    # 3️⃣ 벡터 추출
    idxs = [r["vector_idx"] for r in records]
    vectors = np.array([index.reconstruct(i) for i in idxs]).astype("float32")
    faiss.normalize_L2(vectors)

    # 4️⃣ 쿼리 임베딩
    q_vec = model.encode(query, normalize_embeddings=True).astype("float32").reshape(1, -1)
    faiss.normalize_L2(q_vec)

    # 5️⃣ 부분 인덱스 생성 및 검색
    sub_index = faiss.IndexFlatIP(vectors.shape[1])
    sub_index.add(vectors)

    start = time.time()
    D, I = sub_index.search(q_vec, len(records))
    end = time.time()

    # 6️⃣ threshold 필터링 및 top_k 정렬
    results = [
        (float(D[0][i]), records[I[0][i]]) 
        for i in range(len(records)) if D[0][i] >= threshold
    ]
    results = sorted(results, key=lambda x: x[0], reverse=True)[:top_k]

    # 7️⃣ JSON 반환
    return {
        "results": [
            {"score": s, "title": r["title"], "paragraph": r["paragraph"][:300]}
            for s, r in results
        ],
        "time": round(end - start, 4),
        "count": len(records),
    }

# ================== 서버 실행 ==================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
