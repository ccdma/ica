from seed.mixture import mixed_matrix
from algorithm.easi import EASI
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
np.random.seed(seed=2)

SAMPLE = 4
SERIES = 10000

# 完全にランダムな混合
A = mixed_matrix(SAMPLE)

# 元信号
S = np.array([[np.sin(i/10/(j+1)) for i in range(SERIES)] for j in range(SAMPLE)])
# S = np.array([chebyt_space(i+2, np.random.rand()/2) for i in range(SERIES)]).T

# 混合された信号
X = A @ S # [[x1(0), x1(0.1), x1(0.2)],[x2(0), x2(0.1), x2(0.2)]]のような感じ、右に行くにつれて時間が経過する

easi = EASI(SAMPLE)

YT = []
for x in X.T:
    y = easi.update(x)
    YT.append(y)
Y = np.array(YT).T

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
    ax[i].set_xlim(SERIES-min(500, SERIES),SERIES)
    for j in range(param.shape[0]):
        ax[i].plot(param[j, :])
fig.tight_layout()
plt.show()
