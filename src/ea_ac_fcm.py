from .ac_fcm import AcFCM
import numpy as np


class EAAcFCM:
    def __init__(self, X: np.ndarray, max_clusters: int):
        self.init_clusters = max_clusters
        self.X = X
        self.best_ind = []

    def run(self, iter_max: int) -> AcFCM:
        best_V_XB = np.inf
        for i in range(iter_max):
            pop = [AcFCM(self.X, n_cluster) for n_cluster in range(2, self.init_clusters+1)]
            pop_V_XB = np.array([ind.run() for ind in pop])

            self.best_ind.append(pop[np.argmin(pop_V_XB)])
            if np.min(pop_V_XB) < best_V_XB:
                best_V_XB = np.min(pop_V_XB)
                print(f'New Best Individual (Overall) - {best_V_XB}, Clusters - {self.best_ind[-1].c}')
            print(f'{i}) Population\'s Best Individual: V_XB - {self.best_ind[-1].get_v_xb()}, Clusters - {self.best_ind[-1].c}')

        best_ind_idx = np.argmin([ind.get_v_xb() for ind in self.best_ind])
        return self.best_ind[best_ind_idx]