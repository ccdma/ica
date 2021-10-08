from algorithm.easi import EASI
from eval.dataset import DATADICT, Dataset
from eval.log import Printer
from eval.plot import PlotOption, ReturnmapOption
from eval.operation import inner_matrix
from eval.seed import mixed_matrix
import numpy as np
import matplotlib.pyplot as plt
import sys

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
np.random.seed(seed=4)

def easi_report(dataset: Dataset):

    S = dataset.S
    labels = dataset.labels
    key = dataset.key

    f = open(f"out/easi/{key}.txt", 'w+')
    p = Printer(f, sys.__stdout__)

    SERIES = S.shape[1]
    SAMPLE = S.shape[0]

    p.print(f"SERIES={S.shape[1]}")

    # 完全にランダムな混合
    A = mixed_matrix(SAMPLE)

    # 混合された信号
    X = A @ S # [[x1(0), x1(0.1), x1(0.2)],[x2(0), x2(0.1), x2(0.2)]]のような感じ、右に行くにつれて時間が経過する

    easi = EASI(SAMPLE)

    YT = []
    for x in X.T:
        y = easi.update(x)
        YT.append(y)
    Y = np.array(YT).T

    p.print("A:")
    p.pprint(A)

    p.print("inner(S):")
    p.pprint(inner_matrix(S))
    p.print("inner(X):")
    p.pprint(inner_matrix(X))
    p.print("inner(Y):")
    p.pprint(inner_matrix(Y))

    start_index = SERIES-min(500, SERIES)

    pltOps = [
        PlotOption(S, title="source", labels=labels, start_index=start_index),
        PlotOption(X, title="mixed", start_index=start_index),
        PlotOption(Y, title="reconstruct", start_index=start_index)
    ]
    # グラフの作成
    fig, ax = plt.subplots(len(pltOps), 1)
    for i in range(len(pltOps)):
        pltOps[i].plot(ax[i])
    fig.tight_layout()
    fig.suptitle(f"{key}", x=0.1, y=0.97)
    fig.set_figheight(5)
    fig.set_figwidth(10)
    fig.savefig(f"out/easi/{key}_p.png")

    retOps = [
        ReturnmapOption(S, title="source", labels=labels, start_index=start_index),
        ReturnmapOption(X, title="mixed", start_index=start_index),
        ReturnmapOption(Y, title="recnstruct", start_index=start_index),
    ]
    # リターンマップの作成
    fig, ax = plt.subplots(1, len(retOps))
    for i in range(len(retOps)):
        retOps[i].plot(ax[i])
    fig.tight_layout()
    fig.suptitle(f"{key}", x=0.1, y=0.97)
    fig.set_figheight(5)
    fig.set_figwidth(12)
    fig.savefig(f"out/easi/{key}_r.png")

    f.close()

if __name__ == '__main__':
    # easi_report(DATADICT["isin_10000"])
    # easi_report(DATADICT["sqrtsin_10000"])
    # easi_report(DATADICT["sin2_chebyt2_10000"])
    # easi_report(DATADICT["chebyt2-5_10000"])
    # easi_report(DATADICT["chebyt10_10000"])
    # easi_report(DATADICT["sin+x_10000"])
    # easi_report(DATADICT["sin+2x_10000"])
    # easi_report(DATADICT["A*sin_10000"])
    # easi_report(DATADICT["Adiv2*sin_10000"])    
    # easi_report(DATADICT["Adiv2*sin_5000"])    
    easi_report(DATADICT["A*sqrtsin_10000"])