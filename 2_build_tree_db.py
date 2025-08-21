# build_tree_db.py
import os
import json
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import faiss
import time

MODEL_NAME = 'BM-K/KoSimCSE-roberta-multitask'
DATA_DIR = './장학금'
TREE_FILE = 'tree_db.json'
FAISS_INDEX_FILE = 'faiss.index'

def tree():
    return defaultdict(tree)

def insert_tree(node, path_parts, record):
    for part in path_parts[:-1]:
        node = node[part]
    node.setdefault(path_parts[-1], []).append(record)

def build_tree_db():
    model = SentenceTransformer(MODEL_NAME)
    db_tree = tree()
    all_vectors, all_records = [], []

    for file in os.listdir(DATA_DIR):
        if not file.endswith('.json'):
            continue
        parts = file.replace('.json','').split('-')
        if len(parts)<2:
            print(f"⚠️ 잘못된 파일명: {file}")
            continue
        path = os.path.join(DATA_DIR, file)
        with open(path,'r',encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            vec = model.encode(item['title'], normalize_embeddings=True)
            record = {'title': item['title'], 'paragraph': item['paragraph'], 'vector_idx': len(all_vectors)}
            insert_tree(db_tree, parts, record)
            all_vectors.append(vec)
            all_records.append(record)

    # FAISS 인덱스
    vectors_np = np.array(all_vectors).astype('float32')
    faiss.normalize_L2(vectors_np)
    dim = vectors_np.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors_np)
    faiss.write_index(index, FAISS_INDEX_FILE)

    # 트리 저장
    with open(TREE_FILE,'w',encoding='utf-8') as f:
        json.dump(db_tree,f,ensure_ascii=False,indent=2)

    print("✅ 트리 + 벡터 DB 구축 완료")

if __name__=='__main__':
    build_tree_db()
