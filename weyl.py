from algorithm.fastica_complex import cfast_ica
from eval.log import Printer
import numpy as np
from eval.operation import periodic_correlation
from eval.seed import chebyt_series, mixed_matrix, weyl_series

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True, linewidth=1000)
np.random.seed(seed=4)

p = Printer.with_stdout()

S = np.array([weyl_series(np.random.rand(), np.random.rand(), 1000) for _ in range(4)])
A = mixed_matrix(4)
X = A @ S
Y = cfast_ica(X, _assert=True).Y
l = 2
p.pprint(periodic_correlation(S, l))
p.pprint(periodic_correlation(X, l))
p.pprint(periodic_correlation(Y, l))