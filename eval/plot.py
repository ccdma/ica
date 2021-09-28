import dataclasses as d
import matplotlib.pyplot as plt
import numpy as np
from typing import List 

@d.dataclass
class LabelOption:
    label: str = None
    color: str = None

@d.dataclass
class ReturnmapOption:
    # 描画する行列
    A: np.ndarray
    title: str = None
    start_index: int = 0
    each: List[LabelOption] = None

    def __post_init__(self):
        if self.each == None:
            self.each = []
        for _ in range(self.A.shape[0]-len(self.each)):
            self.each.append(LabelOption())

    def plot(self, ax: plt.Axes):
        ax.set_title(self.title)
        for j in range(self.A.shape[0]): # 各系列
            ax.scatter(self.A[j][self.start_index:-1], self.A[j][self.start_index+1:], s=6, alpha=0.5, label=self.each[j].label)
        if any(map(lambda e : e.label != None, self.each)):
            ax.legend(loc='upper right')

@d.dataclass
class PlotOption:
    # 描画する行列
    A: np.ndarray
    title: str = None
    start_index: int = 0
    end_index: int = None
    labels: List[LabelOption] = None

    def __post_init__(self):
        if self.labels == None:
            self.labels = []
        for _ in range(self.A.shape[0]-len(self.labels)):
            self.labels.append(LabelOption())
        if self.end_index == None:
            self.end_index = self.A.shape[1]

    def plot(self, ax: plt.Axes):
        ax.set_title(self.title)
        for j in range(self.A.shape[0]):
            ax.plot(self.A[j, self.start_index:self.end_index], alpha=0.7, label=self.labels[j].label)
        if any(map(lambda e : e.label != None, self.labels)):
            ax.legend(loc='upper right')