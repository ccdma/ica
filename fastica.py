import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.special import eval_chebyt

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
np.random.seed(seed=1)

SAMPLE = 4
SERIES = 50

# チェビシェフ系列 
# a0: 初期値
def chebyt_space(deg: int, a0: float, length=SAMPLE):
    result = [a0]
    for _ in range(length-1):
        a0 = eval_chebyt(deg, a0)
        result.append(a0)
    return np.array(result) 

# 完全にランダムな混合
A = np.array([[np.random.rand()-0.5 for i in range(SAMPLE)] for j in range(SAMPLE) ])

# 元信号
# S = np.array([[np.sin(i/10/(j+1))+j for i in range(SERIES)] for j in range(SAMPLE)])
S = np.array([chebyt_space(i+2, np.random.rand()/2) for i in range(SERIES)]).T

# 混合された信号
X = A @ S # [[x1(0), x1(0.1), x1(0.2)],[x2(0), x2(0.1), x2(0.2)]]のような感じ、右に行くにつれて時間が経過する

# 中心化を行う（観測点ごとの平均であることに注意）
mean = np.mean(X,axis=1)
X_center = X - np.array([ np.full(SERIES, ave) for ave in mean ]) 

# 固有値分解により、白色化されたX_whitenを計算する
lambdas, P = la.eig(np.cov(X_center)) 
assert np.allclose(np.cov(X_center), P @ np.diag(lambdas) @ P.T) # 固有値分解の検証
for i in reversed(np.where(lambdas < 1.e-12)[0]): # 極めて小さい固有値は削除する
    lambdas = np.delete(lambdas, i, 0)
    P = np.delete(P, i, 1)
Atilda = la.inv(np.sqrt(np.diag(lambdas))) @ P.T # 球面化行列
X_whiten = Atilda @ X_center
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
        B[:,i] = np.average([*map( # 不動点法による更新
            lambda x: g(x @ B[:,i])*x - g2(x @ B[:,i])*B[:,i],
            X_whiten.T
        )], axis=0)
        B[:,i] = B[:,i] - B[:,:i] @ B[:,:i].T @ B[:,i] # 直交空間に射影
        B[:,i] = B[:,i] / la.norm(B[:,i], ord=2) # L2ノルムで規格化
        if 1.0 - 1.e-8 < abs(prevBi @ B[:,i]) < 1.0 + 1.e-8: # （内積1 <=> ほとんど変更がなければ）
            break
    else:
        assert False
assert np.allclose(B @ B.T, np.eye(B.shape[0]), atol=1.e-10) # Bが直交行列となっていることを検証

Y = B.T @ X_whiten

# ica = FastICA(n_components=SERIES, random_state=0)
# _Y = ica.fit_transform(X.T).T * 8

data = [
    [S,"source"],
    [X, "mixed"],
    [Y, "reconstruct"]
]

fig, ax = plt.subplots(len(data), 1)
for i in range(len(data)):
    param = data[i][0]
    title = data[i][1]
    ax[i].set_title(title)
    for j in range(param.shape[0]):
        ax[i].plot(param[j, :])
fig.tight_layout()
plt.show()