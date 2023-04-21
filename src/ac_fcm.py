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
    def absorbtive_criteria(self, inplace: bool = False) -> tuple:
        p, q = self.get_nearest_centroids()
        J_p = self.get_j_fcm(V=self.V[p])
        J_q = self.get_j_fcm(V=self.V[q])
        U, V = self.absorb_c(q, p) if J_p < J_q else self.absorb_c(p, q)

        if inplace:
            self.V = V
            self.U = U
            self.c -= 1

        return U, V