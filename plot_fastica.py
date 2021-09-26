from eval.seed import concat, mixed_matrix
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
S = concat(
    [[np.sin(j/10/(np.sqrt(float(i))+1))+i for j in range(SERIES)] for i in range(SAMPLE)]
)

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