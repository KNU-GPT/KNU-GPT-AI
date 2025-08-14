import os
import json
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('BM-K/KoSimCSE-roberta-multitask')

def load_json_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def preprocess_filename(filename):
    base = filename.replace('.json', '')
    if '장학금' in base:
        base = base.replace('장학금', '')
    base = base.strip('-')
    return base

def main(major, question):
    database_folder = './장학금'
    common_file = '공통-장학금.json'

    all_files = [f for f in os.listdir(database_folder) if f.endswith('.json')]

    common_data = load_json_data(os.path.join(database_folder, common_file))

    file_names = []
    file_name_texts = []

    for filename in all_files:
        if filename == common_file:
            continue
        name_text = preprocess_filename(filename)
        file_names.append(filename)
        file_name_texts.append(name_text)

    # major(과명) 임베딩 (파일명과 비교할 때)
    major_emb = model.encode(major, convert_to_tensor=True)
    file_name_embeddings = model.encode(file_name_texts, convert_to_tensor=True)

    # major와 파일명 유사도 계산
    cos_scores = util.cos_sim(major_emb, file_name_embeddings)[0]

    # 가장 유사도가 높은 파일 하나 선택 (threshold 0.8 이상)
    max_score = cos_scores.max().item()
    if max_score < 0.7:
        print("유사도 0.8 이상인 파일이 없습니다.")
        return

    best_idx = cos_scores.argmax().item()
    best_file = file_names[best_idx]
    print(f"선택된 파일: {best_file} (유사도: {max_score:.3f})")

    # 선택한 파일과 공통 데이터 병합
    best_data = load_json_data(os.path.join(database_folder, best_file))
    merged_data = best_data + common_data

    # 질문 임베딩 (각 항목 title과 비교할 때)
    question_emb = model.encode(question, convert_to_tensor=True)
    item_titles = [item['title'] for item in merged_data]
    item_title_embeddings = model.encode(item_titles, convert_to_tensor=True)

    item_scores = util.cos_sim(question_emb, item_title_embeddings)[0]

    print("\n질문과 title 유사도 0.6 이상인 항목:")
    found = False
    for i, score in enumerate(item_scores):
        if score >= 0.6:
            found = True
            print(f"\n유사도: {score:.3f}")
            print(f"제목: {merged_data[i]['title']}")
            print(f"내용: {merged_data[i]['paragraph'][:300]}...")
            print("---")
    if not found:
        print("유사도 0.6 이상인 항목이 없습니다.")

if __name__ == '__main__':
    major_input = input("과를 입력하세요: ")
    question_input = input("질문을 입력하세요: ")
    main(major_input, question_input)
