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
