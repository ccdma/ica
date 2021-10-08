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

def theta_matrix(P: np.ndarray, l: int):
    N = int(P.shape[1]/2)
    return matrixC(P, l) + matrixC(P, l-N)

def theta_hat_matrix(P: np.ndarray, l: int):
    N = int(P.shape[1]/2)
    return matrixC(P, l) - matrixC(P, l-N)

"""
2つの行ごとの内積を計算し、行列にまとめます
"""
def matrixC(P: np.ndarray, l: int) -> np.ndarray:
    N = int(P.shape[1]/2)
    result = []
    for i in range(P.shape[0]):
        row = []
        for k in range(P.shape[0]):
            if abs(l) >= N:
                row.append(0)
            elif l >= 0:
                _sum = 0
                for n in range(N-1):
                    _sum += np.conjugate(P[i, n+l]) * P[k, n]
                row.append(_sum)
            else:
                _sum = 0
                for n in range(N+1):
                    _sum += np.conjugate(P[i, n]) * P[k, n-l]
                row.append(_sum)
        result.append(row)
    return np.array(result)

"""
行列を縦に結合
"""
def concat(*ndarrays):
    return np.vstack(ndarrays)
