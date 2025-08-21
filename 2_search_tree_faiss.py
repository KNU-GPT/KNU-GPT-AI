import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import time

MODEL_NAME = 'BM-K/KoSimCSE-roberta-multitask'
TREE_FILE = 'tree_db.json'
FAISS_INDEX_FILE = 'faiss.index'

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

def collect_records(node):
    if isinstance(node,list):
        return node
    elif isinstance(node,dict):
        recs=[]
        for v in node.values():
            recs.extend(collect_records(v))
        return recs
    return []

def search_faiss(query, tree, top_k=5):
    depth=int(input("몇 단계까지 탐색? "))
    node = select_node(tree, depth)
    if node is None: return

    records = collect_records(node)
    if not records:
        print("선택 범위에 데이터 없음")
        return

    index = faiss.read_index(FAISS_INDEX_FILE)
    idxs = [r['vector_idx'] for r in records]
    vectors = np.array([index.reconstruct(i) for i in idxs]).astype('float32')
    faiss.normalize_L2(vectors)

    model = SentenceTransformer(MODEL_NAME)
    q_vec = model.encode(query, normalize_embeddings=True).astype('float32').reshape(1,-1)
    faiss.normalize_L2(q_vec)

    start=time.time()
    sub_index = faiss.IndexFlatIP(vectors.shape[1])
    sub_index.add(vectors)
    D,I = sub_index.search(q_vec, top_k)
    end=time.time()

    print(f"\n연산 횟수: {len(records)}, 소요시간: {end-start:.4f}초")
    for score, idx in zip(D[0],I[0]):
        item = records[idx]
        print(f"[{score:.2f}] {item['title']}")
        print(item['paragraph'][:150]+"...\n")


with open('tree_db.json', 'r', encoding='utf-8') as f:
    tree_db = json.load(f)
query = input("검색할 질문 입력: ")
search_faiss(query, tree_db, top_k=5)