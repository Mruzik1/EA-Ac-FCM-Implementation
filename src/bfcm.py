import numpy as np


class BFCM:
    def __init__(self, X: np.ndarray, init_c: int, m: float):
        self.X = X
        self.c = init_c
        self.m = m

        # initializing cluster centroids and a membership matrix + normalizing
        self.U = np.random.normal(size=(X.shape[0], init_c))
        self.U = self.U / np.sum(self.U, axis=1)[:, None]
        self.V = np.random.uniform(np.mean(self.X)-1, np.mean(self.X)+1, size=(init_c, X.shape[1]))

    # getting distance
    def get_d(self, V: np.ndarray = None) -> np.ndarray:
        V = self.V if V is None else V
        return np.sum((self.X[:, None] - V)**2, axis=2)

    # getting a lower obj. func. value J_{fcm}
    def get_j_fcm(self, U: np.ndarray = None, V: np.ndarray = None) -> float:
        U = self.U if U is None else U
        V = self.V if V is None else V
        return np.sum(U**self.m * self.get_d(V))

    # getting an upper obj. func. value V_{XB}
    def get_v_xb(self, U: np.ndarray = None, V: np.ndarray = None, c: int = None) -> float:
        U = self.U if U is None else U
        V = self.V if V is None else V
        c = self.c if c is None else c

        sep = np.min([np.sum((V[i] - V[j])**2)
                    for i in range(c) for j in range(c) if i != j])
        return self.get_j_fcm(U, V) / (sep*self.X.shape[0])

    # optimizing a model, getting a membership matrix and cluster centroids
    def run(self, cmax: int = 200, eps: float = 1e-4, logs_enabled: bool = True) -> tuple:
        for i in range(cmax):
            if logs_enabled:
                print(f'{i}) J_fcm: {self.get_j_fcm()}, V_XB: {self.get_v_xb()}')
            old_V = self.V.copy()

            self.V = np.dot(self.U.T**self.m, self.X) / np.sum(self.U**self.m, axis=0)[:, None]
            self.U = 1 / np.power(self.get_d(), 1 / (self.m - 1))
            self.U = self.U / np.sum(self.U, axis=1)[:, None]

            if np.linalg.norm(self.V - old_V) < eps:
                break
        return self.U, self.V