from eval.plot import PlotOption, ReturnmapOption
from algorithm.easi import EASI
from eval.product import inner_matrix
from eval.seed import chebyt_series, concat, mixed_matrix
import numpy as np
import matplotlib.pyplot as plt
import pprint 

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
np.random.seed(seed=1)

SAMPLE = 4
SERIES = 50000

# 完全にランダムな混合
A = mixed_matrix(SAMPLE)

# 元信号
S = concat(
    [chebyt_series(i+2, np.random.rand()*2-1, SERIES) for i in range(2)],
    [[np.sin(j/10/np.sqrt(i+1)) for j in range(SERIES)] for i in range(SAMPLE-2)]
)

# 混合された信号
X = A @ S # [[x1(0), x1(0.1), x1(0.2)],[x2(0), x2(0.1), x2(0.2)]]のような感じ、右に行くにつれて時間が経過する

easi = EASI(SAMPLE)

YT = []
for x in X.T:
    y = easi.update(x)
    YT.append(y)
Y = np.array(YT).T

pprint.pprint(inner_matrix(S))
pprint.pprint(inner_matrix(X))
pprint.pprint(inner_matrix(Y))

pltOps = [
    PlotOption(S, title="source", start_index=SERIES-min(500, SERIES)),
    PlotOption(X, title="mixed", start_index=SERIES-min(500, SERIES)),
    PlotOption(Y, title="reconstruct", start_index=SERIES-min(500, SERIES))
]
# グラフの作成
fig, ax = plt.subplots(len(pltOps), 1)
for i in range(len(pltOps)):
    pltOps[i].plot(ax[i])
fig.tight_layout()

retOps = [
    ReturnmapOption(S, title="source", start_index=SERIES-min(500, SERIES)),
    ReturnmapOption(X, title="source", start_index=SERIES-min(500, SERIES)),
    ReturnmapOption(Y, title="recnstruct", start_index=SERIES-min(500, SERIES)),
]
# リターンマップの作成
fig, ax = plt.subplots(1, len(retOps))
for i in range(len(retOps)):
    retOps[i].plot(ax[i])
fig.tight_layout()

plt.show()
