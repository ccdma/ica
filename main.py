import numpy as np
from numpy.lib.function_base import average
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.special import eval_chebyt
from sklearn.decomposition import FastICA

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
np.random.seed(seed=1)

SAMPLE = 5
SERIES = 10

# チェビシェフ系列 
# a0: 初期値
def T_space(deg: int, a0: float, length=SAMPLE):
    result = [a0]
    for _ in range(length-1):
        a0 = eval_chebyt(deg, a0)
        result.append(a0)
    return np.array(result) 

# 完全にランダムな混合
A = np.array([[np.random.rand()-0.5 for i in range(SAMPLE)] for j in range(SAMPLE) ])

S = np.array([T_space(i+2, 0.3) for i in range(SERIES)]).T
X = A @ S # [[x1(0), x1(0.1), x1(0.2)],[x2(0), x2(0.1), x2(0.2)]]のような感じ、右に行くにつれて時間が経過する

# 中心化を行う
X_center = X - np.mean(X,axis=0)

# 固有値分解により、白色化されたX_whitenを計算する
lambdas, P = la.eig(np.cov(X_center, rowvar=1)) 
assert np.allclose(np.cov(X_center), P @ np.diag(lambdas) @ P.T)
for i in reversed(np.where(lambdas < 1.e-8)[0]): # 極めて小さい固有値は削除する
    lambdas = np.delete(lambdas, i, 0)
    P = np.delete(P, i, 1)
Atilda = la.inv(np.sqrt(np.diag(lambdas))) @ P.T
X_whiten = Atilda @ X_center # np.cov(X_whiten)は単位行列
assert np.allclose(np.cov(X_whiten, rowvar=1), np.eye(X_whiten.shape[0]), atol=1.e-5)

# ICAに使用する関数g（ここでは４次キュムラントX）
g = lambda bx : bx**3
g2 = lambda bx : 3*bx**2

I = SAMPLE
B = np.eye(I)

# Bを直交行列かつ列ベクトルが大きさ１となるように規格化
for i in range(I):
    if i == 0:
        pass
    else:
        B[:,i] = B[:,i] - B[:,:i] @ B[:,:i].T @ B[:,i]
    B[:,i] = B[:,i] / la.norm(B[:,i], ord=2) # L2ノルムで規格化

# Bの決定(Y=B X_whiten)
for i in range(I):
    k=0
    while True:
        k += 1
        print(k)
        prevBi = B[:,i] 
        B[:,i] = np.average([*map(lambda x : g(x @ B[:,i]) * B[:,i], X_whiten.T)]) \
            - np.average([*map(lambda x : g2(x @ B[:,i]), X_whiten.T)]) * B[:,i]
        if 0.99 < prevBi @ B[:,i] < 1.01: # （内積1 <=> ほとんど変更がなければ）
            break

Y = B.T @ X_whiten

fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax1.plot(S[:, 0])
ax1.plot(Y[:, 0] , label="reconstruct")
ax2 = fig.add_subplot(2,1,2)
ax2.plot(S[:, 1])
ax2.plot(Y[:, 1] , label="reconstruct")

# ica = FastICA(n_components=SERIES, random_state=0)
# Y = ica.fit_transform(X)

# fig = plt.figure()
# ax1 = fig.add_subplot(2,1,1)
# ax1.plot(S[0])
# ax1.plot(np.dot(Y[0]*3, ica.mixing_.T), label="reconstruct")
# ax2 = fig.add_subplot(2,1,2)
# ax2.plot(S[1])
# ax2.plot(np.dot(Y[1]*3, ica.mixing_.T), label="reconstruct")
# # plt.hist(S[0], bins=20)
# # plt.hist(S[0], bins=20)

# # plt.xlim(-1,1)
# # plt.ylim(-1,1)
plt.show()