import numpy as np
from abc import ABC, abstractmethod

class Filter(ABC):
    @abstractmethod
    def predict_and_update(self, z:np.array, dt:float=1.0):
        pass

    @abstractmethod
    def predict(self, dt:float=1.0):
        pass

    @abstractmethod
    def update(self, z:np.array, dt:float=1.0):
        pass

class KalmanFilter(Filter):
    def __init__(self, gt, mes):
        pass

    def predict_and_update(self, z, dt = 1):
        pass

    def predict(self, dt = 1):
        pass

    def update(self, z, dt = 1):
        pass

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