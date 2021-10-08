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

N = int(S.shape[1]/2)

def C(i, k, l, S=S):
    if abs(l) >= N:
        return 0
    elif l >= 0:
        _sum = 0
        for n in range(N-1):
            _sum += np.conjugate(S[i, n+l]) * S[k, n]
        return _sum
    else:
        _sum = 0
        for n in range(N+1):
            _sum += np.conjugate(S[i, n]) * S[k, n-l]
        return _sum

theta = lambda i, k, l : C(i, k, l) + C(i, k, l-N)
p.pprint(theta_matrix(S, 2))
p.pprint(theta(3, 3, 2))
