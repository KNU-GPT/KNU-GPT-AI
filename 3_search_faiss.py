import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import time

TREE_FILE = 'index_tree_db.json'
FAISS_INDEX_FILE = 'faiss.index'
MODEL_NAME = 'BM-K/KoSimCSE-roberta-multitask'

def select_node(tree, depth):
    node = tree
    for i in range(depth):
        options = list(node.keys())
        print(f"[{i+1}단계] 선택 가능한 옵션: {options}")
        choice = input("선택: ").strip()
        if choice not in options:
            print("❌ 잘못 선택함")
            return None
        node = node[choice]
    return node

def print_all_paths(tree, path=None, depth=0):
    if path is None:
        path = []

    if isinstance(tree, dict):
        for key, value in tree.items():
            print("  " * depth + f"└─ {key}")
            print_all_paths(value, path + [key], depth + 1)
    elif isinstance(tree, list):
        print("  " * depth + f"📄 ({len(tree)}개 항목)")


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

def search_faiss(query, tree, top_k=5, threshold=0.6):
    depth = int(input("몇 단계까지 탐색? "))
    node = select_node(tree, depth)
    if node is None:
        return

    records = collect_records(node)
    if not records:
        print("선택 범위에 데이터 없음")
        return

    # FAISS 인덱스 로드
    index = faiss.read_index(FAISS_INDEX_FILE)
    idxs = [r['vector_idx'] for r in records]
    # reconstruct로 벡터 가져오기
    vectors = np.array([index.reconstruct(i) for i in idxs]).astype('float32')
    faiss.normalize_L2(vectors)

    # query 벡터
    model = SentenceTransformer(MODEL_NAME)
    q_vec = model.encode(query, normalize_embeddings=True).astype('float32').reshape(1, -1)
    faiss.normalize_L2(q_vec)

    # 부분 인덱스 생성
    sub_index = faiss.IndexFlatIP(vectors.shape[1])
    sub_index.add(vectors)

    start = time.time()
    D, I = sub_index.search(q_vec, len(records))  # 모든 레코드 대상으로 검색
    end = time.time()

    print(f"\n연산 횟수: {len(records)}, 소요시간: {end-start:.4f}초\n")

    # threshold 이상 필터링 후 top_k만 출력
    results = [(D[0][i], records[I[0][i]]) for i in range(len(records)) if D[0][i] >= threshold]
    results = sorted(results, key=lambda x: x[0], reverse=True)[:top_k]

    if not results:
        print(f"유사도 {threshold} 이상인 항목이 없습니다.")
        return

    for score, item in results:
        print(f"[{score:.2f}] {item['title']}")
        print(item['paragraph'][:300]+"...\n")  # 내용 일부 출력

if __name__ == "__main__":
    with open(TREE_FILE, 'r', encoding='utf-8') as f:
        tree_db = json.load(f)

    print("📚 가능한 전체 선택 트리 구조:")
    print_all_paths(tree_db)

    query = input("검색할 질문 입력: ")
    top_k = int(input("최대 출력 개수 입력: "))
    threshold = float(input("유사도 최소값 입력 (0~1): "))
    search_faiss(query, tree_db, top_k=top_k, threshold=threshold)
