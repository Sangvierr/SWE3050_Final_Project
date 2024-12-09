import numpy as np
from tqdm import tqdm
from time import strftime, localtime
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 20newsgroups 데이터셋 로드 함수
def load_20newsgroups(size_per_class=150, random_state=42):
    print(f'({strftime("%Y-%m-%d %H:%M", localtime())}) 20newsgroups 데이터셋 로드 중...')
    np.random.seed(random_state)
    
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    
    X, y = np.array(newsgroups.data), np.array(newsgroups.target)
    unique_classes = np.unique(y)
    
    sampled_X, sampled_y = [], []

    for class_label in tqdm(unique_classes, desc="샘플링 진행", leave=True):
        class_indices = np.where(y == class_label)[0]
        selected_indices = np.random.choice(class_indices, size=min(size_per_class, len(class_indices)), replace=False)
        sampled_X.extend(X[selected_indices])
        sampled_y.extend(y[selected_indices])
    
    print(f'({strftime("%Y-%m-%d %H:%M", localtime())}) 20newsgroups 데이터셋 로드 완료 >>> 총 {len(sampled_X)}개의 데이터')

    return sampled_X, sampled_y

# IMDB 데이터셋 로드 함수
def load_imdb(sample_size=3000, num_words=3000):
    print(f'({strftime("%Y-%m-%d %H:%M", localtime())}) IMDB 데이터셋 로드 중...')
    
    (X_train_full, y_train_full), (X_test_full, y_test_full) = imdb.load_data(num_words=num_words)
    X = np.concatenate((X_train_full, X_test_full))[:sample_size]
    y = np.concatenate((y_train_full, y_test_full))[:sample_size]

    print(f'({strftime("%Y-%m-%d %H:%M", localtime())}) IMDB 데이터셋 로드 완료 >>> 총 {len(X)}개의 데이터')

    return X, y

# 각 데이터셋 전처리 및 tf-idf 벡터화 함수
def preprocess(X, y, is_imdb=False, random_state=42, max_features=3000):
    
    if is_imdb:
        print(f'({strftime("%Y-%m-%d %H:%M", localtime())}) IMDB 데이터 전처리 및 tf-idf 벡터화 중...')
        X = pad_sequences(X, maxlen=500)
        vectorizer = TfidfVectorizer(max_features=max_features)
        X_tfidf = vectorizer.fit_transform([' '.join(map(str, seq)) for seq in X]).toarray()
        print(f'({strftime("%Y-%m-%d %H:%M", localtime())}) IMDB 데이터 전처리 및 tf-idf 벡터화 완료')

    else:
        print(f'({strftime("%Y-%m-%d %H:%M", localtime())}) 20newsgroups 데이터 전처리 및 tf-idf 벡터화 중...')
        vectorizer = TfidfVectorizer()
        X_tfidf = vectorizer.fit_transform(X).toarray()
        print(f'({strftime("%Y-%m-%d %H:%M", localtime())}) 20newsgroups 데이터 전처리 및 tf-idf 벡터화 완료')
         
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=random_state, stratify=y)
    
    return X_train, X_test, y_train, y_test

def plot_result(result):
    colors = plt.cm.tab10(range(len(result["Model"])))
    fig, axes = plt.subplots(3, 2, figsize=(18, 14))

    # 시각화 대상
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    times = ["Training Time (s)", "Prediction Time (s)"]

    # metrics에 대한 subplot 생성
    for i, metric in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        ax.bar(result["Model"], result[metric], color=colors)
        ax.set_title(metric)
        ax.set_ylabel("Value")
        ax.set_xlabel("Model")
        ax.set_xticks(range(len(result["Model"])))
        ax.set_xticklabels(result["Model"], rotation=45)
        ax.grid(axis='y')

    # times를 일반 스케일로 시각화
    ax = axes[2, 0]
    width = 0.4
    x = range(len(result["Model"]))
    ax.bar(x, result["Training Time (s)"], width=width, label="Training Time (s)", color="skyblue", align='center')
    ax.bar([xi + width for xi in x], result["Prediction Time (s)"], width=width, label="Prediction Time (s)", color="orange", align='center')
    ax.set_title("Training and Prediction Times (Normal Scale)")
    ax.set_ylabel("Time (s)")
    ax.set_xlabel("Model")
    ax.set_xticks([xi + width / 2 for xi in x])
    ax.set_xticklabels(result["Model"], rotation=45)
    ax.grid(axis='y')
    ax.legend()

    # times를 로그 스케일로 시각화
    ax = axes[2, 1]
    ax.bar(x, result["Training Time (s)"], width=width, label="Training Time (s)", color="skyblue", align='center')
    ax.bar([xi + width for xi in x], result["Prediction Time (s)"], width=width, label="Prediction Time (s)", color="orange", align='center')
    ax.set_title("Training and Prediction Times (Log Scale)")
    ax.set_ylabel("Time (s) (log scale)")
    ax.set_yscale("log")
    ax.set_xlabel("Model")
    ax.set_xticks([xi + width / 2 for xi in x])
    ax.set_xticklabels(result["Model"], rotation=45)
    ax.grid(axis='y')
    ax.legend()

    plt.tight_layout()
    plt.show()