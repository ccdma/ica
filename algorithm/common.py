import dataclasses
from typing import List

@dataclasses.dataclass
class Problem:
    X: List[List[int]] # 観測値

@dataclasses.dataclass
class Result:
    Y: List[List[int]] # 分離結果