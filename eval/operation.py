import numpy as np

"""
2つの行ごとの内積を計算し、行列にまとめます
"""
def inner_matrix(P: np.ndarray) -> np.ndarray:
    res = np.eye(P.shape[0], dtype=P.dtype)
    for i in range(P.shape[0]):
        for j in range(P.shape[0]):
            res[i][j] = (P[i]@P[j]) / P.shape[1]
    return res

"""
行列を縦に結合
"""
def concat(*ndarrays):
    return np.vstack(ndarrays)
