import dataclasses as d
import matplotlib.pyplot as plt
import numpy as np

@d.dataclass
class ReturnmapOption:
    # 描画する行列
    A: np.ndarray
    title: str = None
    start_index: int = 0

    def plot(self, ax: plt.Axes):
        ax.set_title(self.title)
        for j in range(self.A.shape[0]): # 各系列
            ax.scatter(self.A[j][self.start_index:-1], self.A[j][self.start_index+1:], s=10, alpha=0.5)

@d.dataclass
class PlotOption:
    # 描画する行列
    A: np.ndarray
    title: str = None
    start_index: int = 0
    end_index: int = None

    def __post_init__(self):
        if self.end_index == None:
            self.end_index = self.A.shape[1]

    def plot(self, ax: plt.Axes):
        ax.set_title(self.title)
        for j in range(self.A.shape[0]):
            ax.plot(self.A[j, self.start_index:self.end_index], alpha=0.7)