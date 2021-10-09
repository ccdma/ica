import numpy.linalg as la
import numpy as np
import dataclasses

from numpy.linalg.linalg import norm

@dataclasses.dataclass
class FastICAResult:
    # Y represents obtained independent data.
    # 
    # EX:
    #   [[y_0(0), y_0(1), y_0(2)]
    #    [y_1(0), y_1(1), y_1(2)]]
    #   s.t. y_point(time)
    Y: np.ndarray

"""
X: represents observed data
[[x1(0), x1(0.1), x1(0.2)],[x2(0), x2(0.1), x2(0.2)]]のような感じ、右に行くにつれて時間が経過する
EX) 
   [[x_0(0), x_0(1), x_0(2)]
    [x_1(0), x_1(1), x_1(2)]]
    s.t. x_point(time) 
"""
def fast_ica(X: np.ndarray, _assert: bool=True) -> FastICAResult:
    SAMPLE, SERIES = X.shape # (観測点数, 観測時間数)

    # 中心化を行う（観測点ごとの平均であることに注意）
    mean = np.mean(X,axis=1)
    X_center = X - np.array([ np.full(SERIES, ave) for ave in mean ]) 

    # 固有値分解により、白色化されたX_whitenを計算する
    lambdas, P = la.eig(np.cov(X_center))
    if _assert:
        assert np.allclose(np.cov(X_center), P @ np.diag(lambdas) @ P.T) # 固有値分解の検証
    for i in reversed(np.where(lambdas < 1.e-12)[0]): # 極めて小さい固有値は削除する
        lambdas = np.delete(lambdas, i, 0)
        P = np.delete(P, i, 1)
    Atilda = la.inv(np.sqrt(np.diag(lambdas))) @ P.T # 球面化行列
    X_whiten = Atilda @ X_center
    if _assert:
        assert np.allclose(np.cov(X_whiten), np.eye(X_whiten.shape[0]), atol=1.e-10) # 無相関化を確認（単位行列）

    # ICAに使用する関数gとその微分g2（ここではgは４次キュムラント）
    g = lambda bx : bx**3
    g2 = lambda bx : 3*(bx**2)

    I = X_whiten.shape[0]
    B = np.array([[np.random.rand()-0.5 for i in range(I)] for j in range(I) ]) # X_whitenからYへの復元行列

    # Bを直交行列かつ列ベクトルが大きさ１となるように規格化
    for i in range(I):
        if i > 0:
            B[:,i] = B[:,i] - B[:,:i] @ B[:,:i].T @ B[:,i] # 直交空間に射影
        B[:,i] = B[:,i] / la.norm(B[:,i], ord=2) # L2ノルムで規格化

    # Bの決定(Y = B.T @ X_whiten)
    for i in range(I):
        for j in range(1000):
            prevBi = B[:,i].copy()
            BiH = np.conjugate(B[:,i].T)
            result = []
            for x in X_whiten.T:
                BiHx = BiH@x
                BiHx2 = np.norm(BiHx, ord=2)
                row = x@BiHx*g(BiHx2) - (g(BiHx2)+BiHx2*g2(BiHx2))@B[:,i]
                result.append(row)
            B[:,i] = np.average(result, axis=0) # 不動点法
            B[:,i] = B[:,i] - B[:,:i] @ B[:,:i].T @ B[:,i] # 直交空間に射影
            B[:,i] = B[:,i] / la.norm(B[:,i], ord=2) # L2ノルムで規格化
            if 1.0 - 1.e-8 < abs(prevBi @ B[:,i]) < 1.0 + 1.e-8: # （内積1 <=> ほとんど変更がなければ）
                break
        else:
            assert False
    if _assert:
        assert np.allclose(B @ B.T, np.eye(B.shape[0]), atol=1.e-10) # Bが直交行列となっていることを検証

    Y = B.T @ X_whiten

    return FastICAResult(Y)

# ica = FastICA(n_components=SERIES, random_state=0)
# _Y = ica.fit_transform(X.T).T * 8