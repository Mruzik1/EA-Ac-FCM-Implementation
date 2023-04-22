import numpy as np
from .bfcm import BFCM


class AcFCM(BFCM):
    def __init__(self, X: np.ndarray, init_c: int, m: float = 2.0):
        super().__init__(X, init_c, m)

    # getting the nearest centroids (to each other)
    def get_nearest_centroids(self) -> tuple:
        D = np.sum((self.V[:, None, :] - self.V[None, :, :])**2, axis=2)
        np.fill_diagonal(D, np.inf)
        return np.unravel_index(np.argmin(D), D.shape)

    # absorbing 2 centroids, remaining the prior centroid
    def absorb_c(self, prior_c: int, absorbed_c: int):
        U = self.U.copy()
        V = self.V.copy()

        V = np.delete(V, absorbed_c, axis=0)
        U[:, prior_c] += U[:, absorbed_c]
        U = np.delete(U, absorbed_c, axis=1)

        return U, V

    # applying absorbtive criteria
    def absorbtive_criteria(self) -> tuple:
        p, q = self.get_nearest_centroids()
        J_p = self.get_j_fcm(V=self.V[p])
        J_q = self.get_j_fcm(V=self.V[q])
        U, V = self.absorb_c(q, p) if J_p < J_q else self.absorb_c(p, q)

        return U, V

    # check and update clusters for step 4
    def update_clusters_4(self, new_U: np.ndarray, new_V: np.ndarray, eps: float) -> bool:
        if self.get_v_xb(new_U, new_V, self.c-1) < self.get_v_xb()+eps:
            self.V = new_V
            self.U = new_U
            self.c -= 1
            return True
        return False

    # check and update clusters for step 5
    def update_clusters_5(self, eps: float) -> bool:
        super().run(logs_enabled=False)
        old_U, old_V = self.U, self.V
        self.U, self.V = self.absorbtive_criteria()
        super().run(logs_enabled=False)

        if self.get_v_xb(old_U, old_V)+eps > self.get_v_xb(c=self.c-1):
            self.c -= 1
            return False
        self.U, self.V = old_U, old_V
        return True
            
    # running an algorithm
    def run(self, eps: float = 1e-4) -> np.ndarray:
        # step 1 + 2
        super().run(logs_enabled=False)
        while self.c > 2:
            print(f'V_XB: {self.get_v_xb()}, clusters: {self.c}')
            
            # step 3 + 4
            if self.update_clusters_4(*self.absorbtive_criteria(), eps=eps):
                continue

            # step 5
            if self.update_clusters_5(eps=eps):
                break
        # step 6
        return self.U, self.V