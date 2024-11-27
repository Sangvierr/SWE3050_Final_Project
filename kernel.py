import numpy as np

# 가중 자카드 커널 함수 정의
def weighted_jaccard_kernel(X1, X2):
    kernel_matrix = np.zeros((len(X1), len(X2)))
    for i, vec1 in enumerate(X1):
        for j, vec2 in enumerate(X2):
            min_sum = np.sum(np.minimum(vec1, vec2))
            max_sum = np.sum(np.maximum(vec1, vec2))
            kernel_matrix[i, j] = min_sum / max_sum if max_sum != 0 else 0
    return kernel_matrix