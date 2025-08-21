import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer, util

MODEL_NAME = 'BM-K/KoSimCSE-roberta-multitask'
TREE_FILE = 'tree_db.json'

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

def search_cosine(query, tree, top_k=5):
    depth=int(input("몇 단계까지 탐색? "))
    node = select_node(tree, depth)
    if node is None: return

    records = collect_records(node)
    if not records: 
        print("선택 범위에 데이터 없음")
        return

    model = SentenceTransformer(MODEL_NAME)
    q_vec = model.encode(query, convert_to_tensor=True)

    start=time.time()
    item_scores=[]
    for item in records:
        t_vec = model.encode(item['title'], convert_to_tensor=True)
        score = util.cos_sim(q_vec, t_vec).item()
        item_scores.append((score,item))
    item_scores.sort(reverse=True,key=lambda x:x[0])
    end=time.time()

    print(f"\n연산 횟수: {len(records)}, 소요시간: {end-start:.4f}초")
    for score,item in item_scores[:top_k]:
        print(f"[{score:.2f}] {item['title']}")
        print(item['paragraph'][:150]+"...\n")


with open('tree_db.json', 'r', encoding='utf-8') as f:
    tree_db = json.load(f)
query = input("검색할 질문 입력: ")
search_cosine(query, tree_db, top_k=5)
