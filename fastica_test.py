from eval.dataset import DATADICT, Dataset
from eval.log import Printer
from eval.plot import PlotOption, ReturnmapOption
from algorithm.fastica import fast_ica
from eval.operation import correlation
from eval.seed import mixed_matrix
import numpy as np
import matplotlib.pyplot as plt
import sys
from collections.abc import Iterable

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
np.random.seed(seed=1)

def fastica_report(dataset: Dataset):

    S = dataset.S
    labels = dataset.labels
    key = dataset.key

    f = open(f"out/fastica/{key}.txt", 'w+')
    p = Printer(f, sys.__stdout__)

    SERIES = S.shape[1]
    SAMPLE = S.shape[0]

    p.print(f"SERIES={S.shape[1]}")

    noise = np.random.normal(0.0,0.01,SERIES)
    S += noise
    # 完全にランダムな混合
    A = mixed_matrix(SAMPLE)

    # 混合された信号
    X = A @ S # [[x1(0), x1(0.1), x1(0.2)],[x2(0), x2(0.1), x2(0.2)]]のような感じ、右に行くにつれて時間が経過する

    res = fast_ica(X, _assert=False)
    Y = res.Y
    X_whiten = res.X_whiten
    X_center = res.X_center

    p.print("A:")
    p.pprint(A)

    p.print("inner(S):")
    p.pprint(correlation(S))
    p.print("inner(X):")
    p.pprint(correlation(X))
    p.print("inner(Y):")
    p.pprint(correlation(Y))

    pltOps = [
        PlotOption(S, title="source", labels=labels, end_index=1000),
        PlotOption(X, title="mixed", end_index=1000),
        # PlotOption(X_center, title="centerized", end_index=1000),
        # PlotOption(X_whiten, title="whiten", end_index=1000),
        PlotOption(Y, title="reconstruct", end_index=1000)
    ]
    # グラフの作成
    fig, ax = plt.subplots(len(pltOps), 1)
    if not isinstance(ax, Iterable):
        ax = [ax]
    for i in range(len(pltOps)):
        pltOps[i].plot(ax[i])
    fig.tight_layout()
    fig.suptitle(f"{key}", x=0.1, y=0.97)
    fig.set_figheight(5)
    fig.set_figwidth(10)
    # fig.savefig(f"out/fastica/{key}_p.png")

    retOps = [
        ReturnmapOption(S, title="source", labels=labels, end_index=1000),
        ReturnmapOption(X, title="mixed", end_index=1000),
        ReturnmapOption(Y, title="recnstruct", end_index=1000),
    ]
    # リターンマップの作成
    fig, ax = plt.subplots(1, len(retOps))
    if not isinstance(ax, Iterable):
        ax = [ax]
    for i in range(len(retOps)):
        retOps[i].plot(ax[i])
    fig.tight_layout()
    fig.suptitle(f"{key}", x=0.1, y=0.97)
    fig.set_figheight(5)
    fig.set_figwidth(12)
    # fig.savefig(f"out/fastica/{key}_r.png")
    plt.show()
    f.close()

if __name__ == '__main__':
    # fastica_report(DATADICT["isin_1000"])
    # fastica_report(DATADICT["sqrtsin_1000"])
    # fastica_report(DATADICT["sin2_chebyt2_1000"])
    fastica_report(DATADICT["chebyt2-5_1000"])
    # fastica_report(DATADICT["chebyt10_1000"])
    # fastica_report(DATADICT["sin+x_1000"])
    # fastica_report(DATADICT["sin+2x_1000"])
    # fastica_report(DATADICT["A*sin_1000"])
    # fastica_report(DATADICT["Adiv2*sin_1000"])    
    # fastica_report(DATADICT["Adiv2*sin_5000"])    
    # fastica_report(DATADICT["2sin_1000"])