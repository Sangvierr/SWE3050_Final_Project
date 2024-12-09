import matplotlib.pyplot as plt
import pandas as pd

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