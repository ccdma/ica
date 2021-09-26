from eval.plot import PlotOption, ReturnmapOption
from algorithm.fastica import fast_ica
from eval.product import inner_matrix
from eval.seed import chebyt_series, concat, mixed_matrix
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
S = concat(
    [chebyt_series(i+2, np.random.rand()*2-1, SERIES) for i in range(2)],
    [[np.sin(j/10/(i+1)) for j in range(SERIES)] for i in range(SAMPLE-2)]
)

# 混合された信号
X = A @ S # [[x1(0), x1(0.1), x1(0.2)],[x2(0), x2(0.1), x2(0.2)]]のような感じ、右に行くにつれて時間が経過する

Y = fast_ica(X, _assert=False).Y

pprint.pprint(inner_matrix(S))
pprint.pprint(inner_matrix(X))
pprint.pprint(inner_matrix(Y))

pltOps = [
    PlotOption(S, title="source"),
    PlotOption(X, title="mixed"),
    PlotOption(Y, title="reconstruct")
]
# グラフの作成
fig, ax = plt.subplots(len(pltOps), 1)
for i in range(len(pltOps)):
    pltOps[i].plot(ax[i])
fig.tight_layout()

retOps = [
    ReturnmapOption(S, title="source"),
    ReturnmapOption(X, title="mixed"),
    ReturnmapOption(Y, title="recnstruct"),
]
# リターンマップの作成
fig, ax = plt.subplots(1, len(retOps))
for i in range(len(retOps)):
    retOps[i].plot(ax[i])
fig.tight_layout()

plt.show()
