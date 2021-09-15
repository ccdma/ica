import dataclasses
import numpy as np
from typing import List

@dataclasses.dataclass
class Problem:
    # represents observed data.
    # 
    # EX:
    #   [[x_0(0), x_0(1), x_0(2)]
    #    [x_1(0), x_1(1), x_1(2)]]
    #   s.t. x_point(time) 
    X: np.ndarray

@dataclasses.dataclass
class Result:
    # represents obtained independent data.
    # 
    # EX:
    #   [[y_0(0), y_0(1), y_0(2)]
    #    [y_1(0), y_1(1), y_1(2)]]
    #   s.t. y_point(time)
    Y: np.ndarray