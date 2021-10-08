from eval.log import Printer
import numpy as np
from eval.operation import periodic_correlation
from eval.seed import chebyt_series, weyl_series

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True, linewidth=1000)
np.random.seed(seed=4)

p = Printer.with_stdout()

S = np.array([weyl_series(np.random.rand(), np.random.rand(), 1000) for _ in range(4)])

p.pprint(periodic_correlation(S, 2))

S = np.array([chebyt_series(i+2, np.random.rand()*2-1, 1000) for i in range(4)])
p.pprint(periodic_correlation(S, 1))