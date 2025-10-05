import os
import json
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer

MODEL_NAME = 'BM-K/KoSimCSE-roberta-multitask'
DATA_DIR = './장학금'
TREE_FILE = 'vector_tree_db.json'

def tree():
    return defaultdict(tree)

def insert_tree(node, path_parts, record):
    for part in path_parts[:-1]:
        node = node[part]
    node.setdefault(path_parts[-1], []).append(record)

def build_tree_json():
    model = SentenceTransformer(MODEL_NAME)
    db_tree = tree()
    all_records = []

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
            vec = model.encode(item['title'], normalize_embeddings=True).tolist()
            record = {
                'title': item['title'],
                'paragraph': item['paragraph'],
                'vector': vec
            }
            insert_tree(db_tree, parts, record)
            all_records.append(record)

    with open(TREE_FILE,'w',encoding='utf-8') as f:
        json.dump(db_tree,f,ensure_ascii=False,indent=2)

    print(f"✅ JSON DB 저장 완료 → {TREE_FILE}")

if __name__=='__main__':
    build_tree_json()
