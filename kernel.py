import numpy as np

# weighted jaccard 커널 정의
def weighted_jaccard_kernel(X1, X2):
    kernel_matrix = np.zeros((len(X1), len(X2)))
    for i, vec1 in enumerate(X1):
        for j, vec2 in enumerate(X2):
            min_sum = np.sum(np.minimum(vec1, vec2))
            max_sum = np.sum(np.maximum(vec1, vec2))
            kernel_matrix[i, j] = min_sum / max_sum if max_sum != 0 else 0
    return kernel_matrix

# 위 커널의 경우 복잡도가 O(N X M X D)
# N: X1의 행 개수, M: X2의 행 개수, D: X1, X2의 열의 개수
# 따라서 N, M, D가 커질수록 계산량이 과도하게 많아짐
