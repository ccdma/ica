import numpy as np

"""
-0.5~+0.5なる混合行列を作成
size: 正方行列のサイズ
"""
def mixed_matrix(size: int):
    return np.array([[np.random.rand()-0.5 for i in range(size)] for j in range(size) ])