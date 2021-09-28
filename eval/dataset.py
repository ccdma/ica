import dataclasses as d
from eval.seed import chebyt_series, concat
from eval.plot import LabelOption
from typing import List
import numpy as np

@d.dataclass
class Dataset: 
    key: str
    S: np.ndarray
    labels: List[LabelOption]
    description: str = None

    def __post_init__(self):
        if(self.description == None):
            self.description = self.key

DATALIST = [
    Dataset(
        key="isin_500",
        S=concat(
            [[np.sin(j/10/(i+4)) for j in range(500)] for i in range(4)]
        ),
        labels=[
            LabelOption(label="sin[x/(10*1)]"),
            LabelOption(label="sin[x/(10*2)]"),
            LabelOption(label="sin[x/(10*3)]"),
            LabelOption(label="sin[x/(10*4)]")
        ],
    ),
    Dataset(
        key="fsin_1000",
        S=concat(
            [[np.sin(j/10/np.sqrt(i+1)) for j in range(1000)] for i in range(4)]
        ),
        labels=[
            LabelOption(label="sin[x/(10√1)]"),
            LabelOption(label="sin[x/(10√2)]"),
            LabelOption(label="sin[x/(10√3)]"),
            LabelOption(label="sin[x/(10√4)]")
        ]
    ),
    Dataset(
        key="sin_chebyt_1000",
        S=concat(
            [chebyt_series(i+2, np.random.rand()*2-1, 1000) for i in range(2)],
            [[np.sin(j/10/np.sqrt(i+1)) for j in range(1000)] for i in range(2)]
        ),
        labels=[
            LabelOption(label="T2"),
            LabelOption(label="T3"),
            LabelOption(label="sin[x/(10√1)]"),
            LabelOption(label="sin[x/(10√2)]")
        ]
    ),
    Dataset(
        key="chebyt",
        S=concat(
            [chebyt_series(i+2, np.random.rand()*2-1, 1000) for i in range(4)],
        ),
        labels=[
            LabelOption(label="T2"),
            LabelOption(label="T3"),
            LabelOption(label="T4"),
            LabelOption(label="T5"),
        ]
    )
]

DATADICT = {}
for data in DATALIST:
    DATADICT[data.key] = data