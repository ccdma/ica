from seed.mixture import mixed_matrix
from seed.chebyt import chebyt_series
from algorithm.fastica import fast_ica
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
np.random.seed(seed=1)

SAMPLE = 4
SERIES = 500

# 完全にランダムな混合
A = mixed_matrix(SAMPLE)

# 元信号
S = np.array([[np.sin(i/10/(np.sqrt(float(j))+1))+j for i in range(SERIES)] for j in range(SAMPLE)])
# S = np.array([chebyt_series(i+2, np.random.rand()/2, SERIES) for i in range(SAMPLE)])

# 混合された信号
X = A @ S # [[x1(0), x1(0.1), x1(0.2)],[x2(0), x2(0.1), x2(0.2)]]のような感じ、右に行くにつれて時間が経過する

result = fast_ica(X, _assert=False)
Y = result.Y

# ica = FastICA(n_components=SERIES, random_state=0)
# _Y = ica.fit_transform(X.T).T * 8

data = [
    [S, "source"],
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