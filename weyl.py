from numpy.random import rand
from eval.product import inner_matrix
from eval.log import Printer
import numpy.linalg as la
import numpy as np

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
np.random.seed(seed=4)

def weyl(n, low_k, delta_k):
    x_raw = n*low_k + delta_k
    x = x_raw - np.floor(x_raw)
    return np.exp(2 * np.pi * 1j * x)

p = Printer.with_sysout()

S = []
for point in range(4):
    s = []
    S.append(s)
    low_k = np.random.rand()
    delta_k = np.random.rand()
    for num in range(1000):
        s.append(weyl(num, low_k, delta_k))
W = np.array(S)
N = int(W.shape[1]/2)

def C(i, k, l, W=W):
    if abs(l) >= N:
        return 0
    elif l >= 0:
        _sum = 0
        for n in range(N-1):
            _sum += np.conjugate(W[i, n+l]) * W[k, n]
        return _sum
    else:
        _sum = 0
        for n in range(N+1):
            _sum += np.conjugate(W[i, n]) * W[k, n-l]
        return _sum

theta = lambda i, k, l : C(i, k, l) + C(i, k, N-l)
p.print(theta(2,1,1))
