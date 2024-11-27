import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

from utils import load_20newsgroups, load_imdb, preprocess, plot_result
from trainer import Trainer


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