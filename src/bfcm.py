import numpy as np


class BFCM:
    def __init__(self, X: np.ndarray, init_c: int, m: float = 2.0):
        self.X = X
        self.c = init_c
        self.m = m

        # initializing cluster centroids and a membership matrix + normalizing
        self.U = np.random.rand(X.shape[0], init_c)
        self.U = self.U / np.sum(self.U, axis=1)[:, None]
        self.V = np.random.rand(init_c, X.shape[1])

    # getting distance
    def get_d(self) -> np.ndarray:
        return np.sum((self.X[:, None] - self.V)**2, axis=2)

    # getting a lower obj. func. value J_{fcm} (normalized)
    def get_j_fcm(self) -> float:
        return np.sum(self.U**self.m * self.get_d()) / self.X.shape[0]

    # getting an upper obj. func. value V_{XB}
    def get_v_xb(self) -> float:
        sep = np.min([np.sqrt(np.sum((self.V[i] - self.V[j]) ** 2))
                    for i in range(self.c) for j in range(self.c) if i != j])
        
        return self.get_j_fcm() / sep

    # optimizing a model, getting a membership matrix and cluster centroids
    def run(self, cmax: int = 200, eps: float = 1e-4) -> tuple:
        for i in range(cmax):
            print(f'{i}) J_fcm: {self.get_j_fcm()}, V_XB: {self.get_v_xb()}')
            old_V = self.V.copy()

            self.V = np.dot(self.U.T**self.m, self.X) / np.sum(self.U**self.m, axis=0)[:, None]
            self.U = 1 / np.power(self.get_d(), 1 / (self.m - 1))
            self.U = self.U / np.sum(self.U, axis=1)[:, None]

            if np.linalg.norm(self.V - old_V) < eps:
                break
        return self.U, self.V