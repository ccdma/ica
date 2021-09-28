from eval.dataset import DATADICT, Dataset
from io import BytesIO
from eval.report import Printer
from eval.plot import LabelOption, PlotOption, ReturnmapOption
from algorithm.fastica import fast_ica
from eval.product import inner_matrix
from eval.seed import chebyt_series, concat, mixed_matrix
import numpy as np
import matplotlib.pyplot as plt
import sys

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
np.random.seed(seed=4)

def fastica_report(dataset: Dataset):

    S = dataset.S
    labels = dataset.labels
    key = dataset.key

    f = open(f"out/fastica/{key}.txt", 'w+')
    p = Printer(f, sys.__stdout__)

    SERIES = S.shape[1]
    SAMPLE = S.shape[0]

    p.print(f"SERIES={S.shape[1]}")

    # 完全にランダムな混合
    A = mixed_matrix(SAMPLE)

    # 混合された信号
    X = A @ S # [[x1(0), x1(0.1), x1(0.2)],[x2(0), x2(0.1), x2(0.2)]]のような感じ、右に行くにつれて時間が経過する

    Y = fast_ica(X, _assert=False).Y

    p.print("A:")
    p.pprint(A)

    p.print("inner(S):")
    p.pprint(inner_matrix(S))
    p.print("inner(X):")
    p.pprint(inner_matrix(X))
    p.print("inner(Y):")
    p.pprint(inner_matrix(Y))

    pltOps = [
        PlotOption(S, title="source", labels=labels),
        PlotOption(X, title="mixed"),
        PlotOption(Y, title="reconstruct")
    ]
    # グラフの作成
    fig, ax = plt.subplots(len(pltOps), 1)
    for i in range(len(pltOps)):
        pltOps[i].plot(ax[i])
    fig.tight_layout()
    fig.suptitle(f"{key}", x=0.1, y=0.97)
    fig.set_figheight(5)
    fig.set_figwidth(10)
    fig.savefig(f"out/fastica/{key}_p.png")

    retOps = [
        ReturnmapOption(S, title="source", labels=labels),
        ReturnmapOption(X, title="mixed"),
        ReturnmapOption(Y, title="recnstruct"),
    ]
    # リターンマップの作成
    fig, ax = plt.subplots(1, len(retOps))
    for i in range(len(retOps)):
        retOps[i].plot(ax[i])
    fig.tight_layout()
    fig.suptitle(f"{key}", x=0.1, y=0.97)
    fig.set_figheight(5)
    fig.set_figwidth(12)
    fig.savefig(f"out/fastica/{key}_r.png")

    f.close()

if __name__ == '__main__':
    fastica_report(DATADICT["isin_1000"])
    fastica_report(DATADICT["sqrtsin_1000"])
    fastica_report(DATADICT["sin2_chebyt2_1000"])
    fastica_report(DATADICT["chebyt2-5_1000"])
