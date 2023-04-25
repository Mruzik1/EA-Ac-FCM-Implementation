from .ac_fcm import AcFCM
import numpy as np


class EAAcFCM:
    def __init__(self, X: np.ndarray, max_clusters: int):
        self.init_clusters = max_clusters
        self.X = X
        self.best_ind = []

    def run(self, iter_max: int, popnum: int) -> AcFCM:
        for i in range(iter_max):
            pop = [AcFCM(self.X, self.init_clusters) for _ in range(popnum)]
            pop_V_XB = np.array([ind.run() for ind in pop])

            self.best_ind.append(pop[np.argmin(pop_V_XB)])
            self.init_clusters = self.best_ind[-1].c
            print(f'{i}) Best Individual: V_XB - {self.best_ind[-1].get_v_xb()}, Clusters - {self.best_ind[-1].c}')

        best_V_XB = np.array([ind.get_v_xb() for ind in self.best_ind])
        return self.best_ind[np.argmin(best_V_XB)]