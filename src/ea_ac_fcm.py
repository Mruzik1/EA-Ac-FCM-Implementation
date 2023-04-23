from .ac_fcm import AcFCM
import numpy as np


class EAAcFCM:
    def __init__(self, X: np.ndarray, max_clusters: int):
        self.clusters = list(range(2, max_clusters+1))
        self.X = X

    def filter_clusters(self, pop: list, pop_V_XB: np.ndarray):
        c_part = len(self.clusters)//3
        worst = {pop[i].c for i in np.argpartition(pop_V_XB, -c_part)[-c_part:]}
        for i in worst:
            if i in self.clusters:
                self.clusters.remove(i)

    def run(self, iter_max: int) -> AcFCM:
        best_ind = []

        for i in range(iter_max):
            pop = [AcFCM(self.X, c) for c in self.clusters]
            pop_V_XB = np.array([ind.run() for ind in pop])
            self.filter_clusters(pop, pop_V_XB)

            best_ind.append(pop[np.argmin(pop_V_XB)])
            print(f'{i}) Best Individual: V_XB - {best_ind[-1].get_v_xb()}, Clusters - {best_ind[-1].c}')

        best_V_XB = np.array([ind.get_v_xb() for ind in best_ind])
        return best_ind[np.argmin(best_V_XB)]