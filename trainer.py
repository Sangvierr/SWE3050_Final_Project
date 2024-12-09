import time
import pandas as pd
from time import strftime, localtime
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from kernel import weighted_jaccard_kernel

class Trainer:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.target_models = {
            "Weighted Jaccard SVM": weighted_jaccard_kernel
        }
        self.compare_models = {
            "Linear SVM": SVC(kernel='linear'),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "KNN(K=3)": KNeighborsClassifier(n_neighbors=3),
            "KNN(K=5)": KNeighborsClassifier(n_neighbors=5),
            "Naive Bayes": MultinomialNB(),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "SGD": SGDClassifier(random_state=42)
        }

    @staticmethod
    def _evaluate_model(y_test, y_pred, model_name, training_time, prediction_time):
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        return {
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Training Time (s)": training_time,
            "Prediction Time (s)": prediction_time
        }

    def _train_target_models(self):
        results = []
        for kernel_name, kernel_func in self.target_models.items():
            print(f'▶ {kernel_name} 학습 시작...')
            if not callable(kernel_func):
                raise ValueError(f"Kernel function {kernel_name} is not callable.")
            start_time = time.time()
            K_train = kernel_func(self.X_train, self.X_train)
            K_test = kernel_func(self.X_test, self.X_train)
            model = SVC(kernel='precomputed')
            model.fit(K_train, self.y_train)
            training_time = time.time() - start_time

            pred_start_time = time.time()
            y_pred = model.predict(K_test)
            prediction_time = time.time() - pred_start_time

            results.append(self._evaluate_model(self.y_test, y_pred, kernel_name, training_time, prediction_time))
            print(f'▶ {kernel_name} 학습 완료 (Training Time: {training_time:.2f}s)')
        return results

    def _train_compare_models(self):
        results = []
        for model_name, model in self.compare_models.items():
            print(f'▶ {model_name} 학습 시작...')
            start_time = time.time()
            model.fit(self.X_train, self.y_train)
            training_time = time.time() - start_time

            pred_start_time = time.time()
            y_pred = model.predict(self.X_test)
            prediction_time = time.time() - pred_start_time

            results.append(self._evaluate_model(self.y_test, y_pred, model_name, training_time, prediction_time))
            print(f'▶ {model_name} 학습 완료 (Training Time: {training_time:.2f}s)')
        return results

    def train_all_models(self, only_target=False):
        print(f'({strftime("%Y-%m-%d %H:%M", localtime())}) 타겟 모델 학습 시작...')
        target_results = self._train_target_models()
        print(f'({strftime("%Y-%m-%d %H:%M", localtime())}) 타겟 모델 학습 종료')
        
        # 비교 모델 학습 (조건부 실행)
        if not only_target:
            print(f'({strftime("%Y-%m-%d %H:%M", localtime())}) 비교 모델 학습 시작...')
            compare_results = self._train_compare_models()
            print(f'({strftime("%Y-%m-%d %H:%M", localtime())})) 비교 모델 학습 종료')
            
            # 타겟 모델 + 비교 모델 결과 반환
            return pd.DataFrame(target_results + compare_results) 
        
        # 타겟 모델 결과만 반환
        return pd.DataFrame(target_results)