import numpy as np

"""
2つの列ごとの内積を計算し、行列にまとめます
"""
def inner_matrix(P: np.ndarray) -> np.ndarray:
    res = np.eye(P.shape[0])
    for i in range(P.shape[0]):
        for j in range(P.shape[0]):
            res[i][j] = (P[i]@P[j]) / P.shape[1]
    return res