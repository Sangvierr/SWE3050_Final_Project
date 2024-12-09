import warnings
warnings.filterwarnings('ignore')

import pandas as pd

from utils import load_20newsgroups, load_imdb, preprocess
from viz import plot_experiment_v1, plot_experiment_v2
from trainer import Trainer


# 20newsgroups 실험
def main1():
    print("\n[실험 1] 20newsgroups 데이터셋을 활용한 다중분류 성능 비교\n")
    
    X, y = load_20newsgroups(size_per_class=250, random_state=42) # 250개씩 20개 클래스 => 5000개
    X_train, X_test, y_train, y_test = preprocess(X, y)

    # 학습 시작
    trainer = Trainer(X_train, X_test, y_train, y_test)
    result = trainer.train_all_models()
    
    print('\n[실험 1] 다중분류 성능 비교 결과\n')
    print(result)
    
    return result

# IMDB 실험
def main2():
    print("\n[실험 2] IMDB 데이터셋을 활용한 이진분류 성능 비교\n")
    
    X, y = load_imdb(sample_size=5000, num_words=5000) # 5000개
    X_train, X_test, y_train, y_test = preprocess(X, y, is_imdb=True)

    # 학습 시작
    trainer = Trainer(X_train, X_test, y_train, y_test)
    result = trainer.train_all_models()
    
    print('\n[실험 2] 이진분류 성능 비교 결과\n')
    print(result)
    
    return result

# IMDB 데이터셋의 크기와 num_words에 따른 성능 비교 실험
def main3():
    print("\n[실험 3] IMDB 데이터셋의 크기와 num_words에 따른 성능 비교")
    
    sample_sizes = [500, 1000, 2000, 3000, 4000, 5000]
    num_words_list = [1000, 2000, 3000, 4000, 5000]
    
    result = []
    total_steps = len(sample_sizes) * len(num_words_list)
    
    for i, sample_size in enumerate(sample_sizes):
        for j, num_words in enumerate(num_words_list):
            step_count = i * len(num_words_list) + j + 1
            print(f"\n[Experiment 3 | Step {step_count}/{total_steps}] sample_size={sample_size}, num_words={num_words}")
            
            X, y = load_imdb(sample_size=sample_size, num_words=num_words)
            X_train, X_test, y_train, y_test = preprocess(X, y, is_imdb=True)
            
            # 학습 시작
            trainer = Trainer(X_train, X_test, y_train, y_test)
            model_result = trainer.train_all_models(only_target=True)
            
            model_result['sample_size'] = sample_size
            model_result['num_words'] = num_words
            
            result.append(model_result)
    
    print("\n[실험 3] IMDB 데이터셋의 크기와 num_words에 따른 성능 비교 결과\n")
    print(result)
    
    return pd.concat(result, ignore_index=True)

    

if __name__ == '__main__':
    result1 = main1()
    plot_experiment_v1(result1)
    result1.to_csv('result1.csv', index=False)
    
    result2 = main2()
    plot_experiment_v1(result2)
    result2.to_csv('result2.csv', index=False)
    
    result3 = main3()
    plot_experiment_v2(result3)
    result3.to_csv('result3.csv', index=False)