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
    def __init__(self, gt, mes):
        pass

    def predict_and_update(self, z, dt = 1):
        pass

    def predict(self, dt = 1):
        pass

    def update(self, z, dt = 1):
        pass

class ParticleFilter(Filter):
    def __init__(self, gt, mes):
        pass

    def predict_and_update(self, z, dt = 1):
        pass

    def predict(self, dt = 1):
        pass

    def update(self, z, dt = 1):
        pass