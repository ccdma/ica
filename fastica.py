from algorithm.common import Problem
from algorithm.fastica import FastICA
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

result = FastICA(Problem(X), _assert=False)
Y = result.Y

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