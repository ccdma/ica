from numpy.random import rand
from eval.log import Printer
import numpy as np
import numpy.linalg as la
from eval.operation import theta_matrix

from eval.seed import weyl_series

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True, linewidth=1000)
np.random.seed(seed=4)

p = Printer.with_sysout()

S = np.array([weyl_series(np.random.rand(), np.random.rand(), 1000) for _ in range(4)])

p.pprint(theta_matrix(S, 2))
