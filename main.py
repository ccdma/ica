import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.special import eval_chebyt
from sklearn.decomposition import FastICA

np.random.seed(seed=1)
SAMPLE = 20
SERIES = 5

def T_space(deg: int, a0: float):
    result = [a0]
    for _ in range(SAMPLE-1):
        a0 = eval_chebyt(deg, a0)
        result.append(a0)
    return np.array(result)

A = np.array([[np.random.rand()-0.5 for i in range(SAMPLE)] for j in range(SAMPLE) ])

S = np.array([T_space(i+5, 0.3) for i in range(SERIES)]).T
X = np.dot(A, S) # [[x1(0), x1(0.1), x1(0.2)],[x2(0), x2(0.1), x2(0.2)]]のような感じ、右に行くにつれて時間が経過する

# 中心化を行う
X_center = X - np.mean(X)

# 固有値分解により、白色化されたX_whitenを計算する
lambdas, P = la.eig(np.cov(X_center))
Atilda = np.dot( la.inv(np.sqrt(np.diag(lambdas))) , P.T )
X_whiten = np.dot(Atilda, X_center) # np.cov(X_whiten)は対角行列なはず

ica = FastICA(n_components=SERIES, random_state=0)
Y = ica.fit_transform(X)

fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax1.plot(S[0])
ax1.plot(np.dot(Y[0]*3, ica.mixing_.T), label="reconstruct")
ax2 = fig.add_subplot(2,1,2)
ax2.plot(S[1])
ax2.plot(np.dot(Y[1]*3, ica.mixing_.T), label="reconstruct")
# plt.hist(S[0], bins=20)
# plt.hist(S[0], bins=20)

# plt.xlim(-1,1)
# plt.ylim(-1,1)
plt.show()