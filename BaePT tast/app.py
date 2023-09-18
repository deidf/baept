from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import string
import os  # 파일 경로 조작을 위한 모듈

app = Flask(__name__)

# 데이터 로드 및 전처리
file_path = "C:/Users/xo060/Desktop/BaePT tast/BaePT(t).xlsx"
df = pd.read_excel(file_path, engine='openpyxl')
df.fillna('', inplace=True)
df.replace([float('inf'), float('-inf')], '', inplace=True)
question_list = df['Q'].tolist()

vectorizer = CountVectorizer()
sentence_vectors = vectorizer.fit_transform(question_list)
sentence_vectors = sentence_vectors.toarray()

# 'save' 폴더 경로 설정
save_folder = "C:/Users/xo060/Desktop/BaePT tast/save"

@app.route('/', methods=['GET', 'POST'])
def index():
    answer = ''
    if request.method == 'POST':
        new_sentence = request.form['user_input']
        new_sentence_vectors = vectorizer.transform([new_sentence])
        new_sentence_vectors = new_sentence_vectors.toarray()

        similarity_scores = cosine_similarity(sentence_vectors, new_sentence_vectors)
        max_idx = similarity_scores.argmax()

        if max_idx == 0:
            answer = "다시 입력해 주세요."
        else:
            answer = df['N'].tolist()[max_idx]

        # 파일 저장 및 로그 기록
        filename = ''.join(random.choices(string.ascii_letters, k=30)) + ".txt"
        file_path = os.path.join(save_folder, filename)  # 'save' 폴더 내에 파일 경로 생성
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f'Q: {new_sentence}\nN: {answer}')

    return render_template('BaePT.html', answer=answer)

if __name__ == '__main__':
    app.run()
