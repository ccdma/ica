import numpy as np

class EASI:
    def __init__(self, size: int, mu=0.001953125, g=lambda x:-np.tanh(x)):
        self.B = np.array([[np.random.rand()-0.5 for i in range(size)] for j in range(size) ]) # 復元行列
        self._g = g # 更新関数
        self._mu = mu # 更新時パラメータ
        self._size = size # 観測点数
    
    """
    新しく観測したxを更新します
    x: ndarray (self.size長ベクトル)

    returns: 復元ベクトル
    """
    def update(self, x: np.ndarray) -> np.ndarray:
        y = np.array([self.B @ x]).T
        V = y @ y.T - np.eye(self._size) + self._g(y) @ y.T - y @ self._g(y).T
        self.B = self.B - self._mu * V @ self.B
        return y[:,0]
