from io import BytesIO
from eval.report import Printer
from eval.plot import EachOption, PlotOption, ReturnmapOption
from algorithm.fastica import fast_ica
from eval.product import inner_matrix
from eval.seed import chebyt_series, concat, mixed_matrix
import numpy as np
import matplotlib.pyplot as plt
import sys

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
np.random.seed(seed=2)

def main(S, desc, title):

    f = open(f"out/fastica/{title}.txt", 'w+')
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
        PlotOption(S, title="source", end_index=500, each=desc),
        PlotOption(X, title="mixed", end_index=500),
        PlotOption(Y, title="reconstruct", end_index=500)
    ]
    # グラフの作成
    fig, ax = plt.subplots(len(pltOps), 1)
    for i in range(len(pltOps)):
        pltOps[i].plot(ax[i])
    fig.tight_layout()
    fig.suptitle(f"{title}", x=0.1, y=0.97)
    fig.set_figheight(5)
    fig.set_figwidth(10)
    fig.savefig(f"out/fastica/{title}_p.png")

    retOps = [
        ReturnmapOption(S, title="source", each=desc),
        ReturnmapOption(X, title="mixed"),
        ReturnmapOption(Y, title="recnstruct"),
    ]
    # リターンマップの作成
    fig, ax = plt.subplots(1, len(retOps))
    for i in range(len(retOps)):
        retOps[i].plot(ax[i])
    fig.tight_layout()
    fig.suptitle(f"{title}", x=0.1, y=0.97)
    fig.set_figheight(5)
    fig.set_figwidth(12)
    fig.savefig(f"out/fastica/{title}_r.png")

    f.close()

if __name__ == '__main__':
    SERIES=1000

    # main()
    S = concat(
        [[np.sin(j/10/np.sqrt(i+1)) for j in range(SERIES)] for i in range(4)]
    )
    desc = [
        EachOption(label="sin[x/(10√1)]"),
        EachOption(label="sin[x/(10√2)]"),
        EachOption(label="sin[x/(10√3)]"),
        EachOption(label="sin[x/(10√4)]")
    ]
    title = "fsin"
    main(S, desc, title)

    S = concat(
        [[np.sin(j/10/(i+4)) for j in range(500)] for i in range(4)]
    )
    desc = [
        EachOption(label="sin[x/(10*1)]"),
        EachOption(label="sin[x/(10*2)]"),
        EachOption(label="sin[x/(10*3)]"),
        EachOption(label="sin[x/(10*4)]")
    ]
    title = "isin"
    main(S, desc, title)

    S = concat(
        [chebyt_series(i+2, np.random.rand()*2-1, SERIES) for i in range(4)],
    )
    desc = [
        EachOption(label="T2"),
        EachOption(label="T3"),
        EachOption(label="T4"),
        EachOption(label="T5"),
    ]
    title = "chebyt"
    main(S, desc, title)

    S = concat(
        [chebyt_series(i+2, np.random.rand()*2-1, SERIES) for i in range(2)],
        [[np.sin(j/10/np.sqrt(i+1)) for j in range(SERIES)] for i in range(2)]
    )
    desc = [
        EachOption(label="T2"),
        EachOption(label="T3"),
        EachOption(label="sin[x/(10√1)]"),
        EachOption(label="sin[x/(10√2)]")
    ]
    title = "sin_chebyt"
    main(S, desc, title)

    S = concat(
        [chebyt_series(i+2, np.random.rand()*2-1, SERIES) for i in range(4)],
    )
    desc = [
        EachOption(label="T2"),
        EachOption(label="T3"),
        EachOption(label="T4"),
        EachOption(label="T5"),
    ]
    title = "chebyt"
    main(S, desc, title)
