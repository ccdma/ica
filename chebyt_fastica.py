
from eval.product import inner_matrix
from eval.seed import chebyt_series, mixed_matrix
from algorithm.fastica import fast_ica
import numpy as np
import matplotlib.pyplot as plt
import pprint 

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
np.random.seed(seed=1)

SAMPLE = 3
SERIES = 1000

# 完全にランダムな混合
A = mixed_matrix(SAMPLE)

# 元信号
S = np.array([chebyt_series(i+2, np.random.rand()/2, SERIES) for i in range(SAMPLE)])

# 混合された信号
X = A @ S # [[x1(0), x1(0.1), x1(0.2)],[x2(0), x2(0.1), x2(0.2)]]のような感じ、右に行くにつれて時間が経過する

result = fast_ica(X, _assert=False)
Y = result.Y

data = [
    [S, "source"],
    [X, "mixed"],
    [Y, "reconstruct"]
]

pprint.pprint(inner_matrix(S))
pprint.pprint(inner_matrix(X))
pprint.pprint(inner_matrix(Y))

fig, ax = plt.subplots(1, len(data))
for i in range(len(data)):
    ax[i].set_title(data[i][1])
    ax[i].scatter(data[i][0][0][0:-1], data[i][0][0][1:], s=10, c="blue", alpha=0.5)
    ax[i].scatter(data[i][0][1][0:-1], data[i][0][1][1:], s=10, c="green", alpha=0.5)
    ax[i].scatter(data[i][0][2][0:-1], data[i][0][2][1:], s=10, c="red", alpha=0.5)
fig.tight_layout()
plt.show()

