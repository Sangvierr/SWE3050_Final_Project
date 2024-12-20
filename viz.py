import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

# 실험 1, 2 결과 시각화 함수
def plot_experiment_v1(result):
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
    
# 실험 3 결과 시각화 함수
def plot_experiment_v2(result):
    assert isinstance(result, pd.DataFrame), "입력 인자는 pandas.DataFrame이어야 합니다."
    
    # num_words 고유값 추출
    num_words_list = sorted(result['num_words'].unique())
    
    # 마커 스타일 설정
    markers = ['o', 's', 'D', '^', 'v', '<', '>']
    marker_dict = {num_words: markers[i % len(markers)] for i, num_words in enumerate(num_words_list)}
    
    # 시각화할 메트릭 정의
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "Training Time (s)", "Prediction Time (s)"]
    
    # 서브플롯 설정
    fig, axes = plt.subplots(3, 2, figsize=(16, 9))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for num_words in num_words_list:
            subset = result[result['num_words'] == num_words].sort_values(by='sample_size')
            ax.plot(subset['sample_size'], subset[metric],
                    label=f"Num Words: {num_words}",
                    marker=marker_dict[num_words],
                    linestyle='-')
        
        ax.set_title(metric)
        ax.set_xlabel("Sample Size")
        ax.set_ylabel(metric)
        ax.grid(True, linestyle='--')
        ax.tick_params(axis='x')
        ax.tick_params(axis='y')
    
    # 범례 설정: 플롯과 가깝게 표시
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, 
               title="Num Words",
               loc='upper center',
               ncol=len(num_words_list))
    
    # 전체 레이아웃 조정
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
# IMDB 데이터 시각화
def plot_embeddings_imdb(X_tfidf, X_kernel, y):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    reducers = {
        "PCA": PCA(n_components=2),
        "t-SNE": TSNE(n_components=2, perplexity=30, random_state=42),
        "UMAP": UMAP(n_neighbors=15, min_dist=0.1, random_state=42),
    }

    for i, (name, reducer) in enumerate(reducers.items()):
        # TF-IDF embeddings
        Z_tfidf = reducer.fit_transform(X_tfidf)
        axes[0, i].scatter(Z_tfidf[y == 0, 0], Z_tfidf[y == 0, 1], c="blue", s=3, label="Class 0")
        axes[0, i].scatter(Z_tfidf[y == 1, 0], Z_tfidf[y == 1, 1], c="red", s=3, label="Class 1")
        axes[0, i].set_title(f"TF-IDF Embeddings Visualization - {name}")

        # Kernel embeddings
        Z_kernel = reducer.fit_transform(X_kernel)
        axes[1, i].scatter(Z_kernel[y == 0, 0], Z_kernel[y == 0, 1], c="blue", s=3, label="Class 0")
        axes[1, i].scatter(Z_kernel[y == 1, 0], Z_kernel[y == 1, 1], c="red", s=3, label="Class 1")
        axes[1, i].set_title(f"Weighted Jacard Kernel Embeddings Visualization - {name}")

    plt.tight_layout()
    plt.show()
 
# 20newsgroups 데이터 시각화   
def plot_embeddings_20news(X_tfidf, X_kernel, y, unique_classes):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    reducers = {
        "PCA": PCA(n_components=2),
        "t-SNE": TSNE(n_components=2, perplexity=30, random_state=42),
        "UMAP": UMAP(n_neighbors=15, min_dist=0.1, random_state=42),
    }

    for i, (name, reducer) in enumerate(reducers.items()):
        # TF-IDF embeddings
        Z_tfidf = reducer.fit_transform(X_tfidf)
        for class_label in unique_classes:
            class_indices = (y == class_label)
            axes[0, i].scatter(Z_tfidf[class_indices, 0], Z_tfidf[class_indices, 1], s=4, label=f"Class {class_label}")
        axes[0, i].set_title(f"TF-IDF Embeddings Visualization - {name}")
        axes[0, i].legend(loc="lower right", fontsize=6, bbox_to_anchor=(1.05, 0.05))

        # Kernel embeddings
        Z_kernel = reducer.fit_transform(X_kernel)
        for class_label in unique_classes:
            class_indices = (y == class_label)
            axes[1, i].scatter(Z_kernel[class_indices, 0], Z_kernel[class_indices, 1], s=4, label=f"Class {class_label}")
        axes[1, i].set_title(f"Weighted Jaccard Kernel Embeddings Visualization - {name}")
        axes[1, i].legend(loc="lower right", fontsize=6, bbox_to_anchor=(1.05, 0.05))

    plt.tight_layout()
    plt.show()