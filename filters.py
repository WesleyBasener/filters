import numpy as np
from abc import ABC, abstractmethod

class Filter(ABC):
    @abstractmethod
    def predict_and_update(self, z:np.array):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def update(self, z:np.array):
        pass

class KalmanFilter(Filter):
    def __init__(self, gt, mes, init_x:np.array=None, init_P:np.array=None):
        self.F = gt.F
        self.Q = gt.Q
        self.R = mes.R
        self.H = mes.H

        if init_x is None:
            self.x = gt.ground_truth[0]
        else:
            self.x = init_x

        if init_P is None:
            self.P = np.identity(len(self.x))
        else:
            self.P = init_P

    def predict_and_update(self, z):
        self.predict()
        self.update(z)

    def predict(self):
        self.x_pred = self.F @ self.x

        self.P_pred = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        y = z - self.H @ self.x

        S = self.H @ self.P_pred @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x_pred + K @ y
        self.P = (np.identity(self.P.shape[0]) - K @ self.H) @ self.P_pred

class ExtendedKalmanFilter(Filter):
    def __init__(self, gt, mes):
        pass

    def predict_and_update(self, z, dt = 1):
        pass

    def predict(self, dt = 1):
        pass

    def update(self, z, dt = 1):
        pass
    
class UnscentedKalmanFilter(Filter):

    def __init__(self, gt, mes, alpha:float, beta:float, k:float=None, init_x:np.array=None, init_P:np.array=None):
        self.F = gt.F
        self.Q = gt.Q
        self.R = mes.R
        self.measurement_method = mes.measurement_method

        if init_x is None:
            self.x = gt.ground_truth[0]
        else:
            self.x = init_x

        if init_P is None:
            self.P = np.identity(len(self.x))
        else:
            self.P = init_P

        self.n = len(self.x)
        if k is None:
            self.k = 3 - self.n
        else:
            self.k = k
        self.alpha = alpha
        self.beta = beta
        self.lam = (self.alpha**2)*(self.n + self.k) - self.n

        W_c = [self.lam/(self.n + self.lam) + 1 - self.alpha**2 + self.beta]
        W_m = [self.lam/(self.n + self.lam)]

        for _ in range(2*self.n):
            W_c.append(1/(2*(self.n + self.lam)))
            W_m.append(1/(2*(self.n + self.lam)))

        self.W_c = np.array(W_c)
        self.W_m = np.array(W_m)

    def predict_and_update(self, z):
        self.predict()
        self.update(z)

    def predict(self):
        X = self.get_sigmas()
        self.Y = X @ self.F.T

        self.x_pred, self.P_pred = self.unscented_transform(self.Y, self.Q)

    def update(self, z):
        Z = self.measure(self.Y)

        mean_z, P_z = self.unscented_transform(Z, self.R)

        y = z - mean_z

        K = np.zeros((len(self.x_pred), len(mean_z)))
        for i in range(len(self.W_c)):
            y_std = self.Y[i] - self.x_pred
            z_std = Z[i] - mean_z
            K += self.W_c[i] * np.outer(y_std, z_std)
        K = K @ np.linalg.inv(P_z)

        self.x = self.x_pred + K @ y
        self.P = self.P_pred - K @ P_z @ K.T

    def get_sigmas(self):
        S = np.linalg.cholesky((self.lam+self.n)*self.P, upper=True)
        X = np.zeros((1+2*self.n, self.n))
        
        X[0] = self.x
        for i in range(self.n):
            X[i + 1] = self.x + S[i]
            X[self.n + i + 1] = self.x - S[i]

        return X

    def measure(self, X):
        Z = np.zeros((X.shape[0], self.R.shape[0]))
        for i in range(X.shape[0]):
            Z[i] = self.measurement_method(X[i])
        return Z

    def unscented_transform(self, X, S):
        
        x_mean = self.W_m @ X

        covariance = np.zeros((len(x_mean), len(x_mean)))
        for i in range(len(self.W_c)):
            y = X[i] - x_mean 
            covariance += self.W_c[i]*np.outer(y, y)
        covariance += S

        return x_mean, covariance

class ParticleFilter(Filter):
    def __init__(self, gt, mes):
        pass

    def predict_and_update(self, z, dt = 1):
        pass

    def predict(self, dt = 1):
        pass

    def update(self, z, dt = 1):
        pass