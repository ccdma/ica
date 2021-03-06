from scipy.special import eval_chebyt
import numpy as np

"""
チェビシェフ系列を生成
deg: チェビシェフ多項式の次数
a0: 初期値
length: 系列の長さ
"""
def chebyt_series(deg: int, a0: float, length: int) -> np.ndarray:
    result = [a0]
    for _ in range(length-1):
        a0 = eval_chebyt(deg, a0)
        result.append(a0)
    return np.array(result) 

"""
ワイル系列を生成
return ndarray(dtype=complex)
"""
def weyl_series(low_k: float, delta_k: float, length: int) -> np.ndarray:
    result = []
    for n in range(length):
        x_raw = n*low_k + delta_k
        x = x_raw - np.floor(x_raw)
        result.append(np.exp(2 * np.pi * 1j * x))
    return np.array(result)

"""
-0.5~+0.5なる混合行列を作成
size: 正方行列のサイズ
"""
def mixed_matrix(size: int) -> np.ndarray:
    return np.array([[np.random.rand()-0.5 for i in range(size)] for j in range(size) ])


