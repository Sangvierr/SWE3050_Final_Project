import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

from datasets import load_20newsgroups, load_imdb, preprocess
from trainer import Trainer

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


# 20newsgroups 실험
def main1():
    print("[실험 1] 20newsgroups 데이터셋을 활용한 다중분류 성능 비교")
    
    X, y = load_20newsgroups(size_per_class=250, random_state=42) # 250개씩 20개 클래스 => 5000개
    X_train, X_test, y_train, y_test = preprocess(X, y)

    # 학습 시작
    trainer = Trainer(X_train, X_test, y_train, y_test)
    result = trainer.train_all_models()
    
    print('[실험 1] 다중분류 성능 비교 결과')
    print(result)
    
    return result

# IMDB 실험
def main2():
    print("[실험 2] IMDB 데이터셋을 활용한 이진분류 성능 비교")
    
    X, y = load_imdb(sample_size=5000, num_words=5000) # 5000개
    X_train, X_test, y_train, y_test = preprocess(X, y, is_imdb=True)

    # 학습 시작
    trainer = Trainer(X_train, X_test, y_train, y_test)
    result = trainer.train_all_models()
    
    print('[실험 2] 이진분류 성능 비교 결과')
    print(result)
    
    return result

if __name__ == '__main__':
    result1 = main1()
    plot_result(result1)
    result1.to_csv('result1.csv', index=False)
    
    result2 = main2()
    plot_result(result2)
    result2.to_csv('result2.csv', index=False)
    
    #result3 = main3()
    #plot_result(result3)
    #result3.to_csv('result3.csv', index=False)