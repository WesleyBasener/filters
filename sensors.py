import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Callable

from utils import rbe_to_cart, cart_to_rbe
 
@dataclass
class Measurement:
    
    measurements:np.array
    measurements_gt:np.array
    R:np.array
    H:np.array
    measurement_method:Callable
    sensor:str

class Sensor(ABC):
    @abstractmethod
    def measure_ground_truths(self, gt):
        pass

    def measure(sekf, x):
        pass

class XYZSensor(Sensor):

    def __init__(self, dim=3, num_dirs_gt=1, num_dirs_mes=0, cov=None):
        
        if cov == None:
            cov = np.ones(num_dirs_mes+1)

        self.R = np.kron(np.identity(dim*(1+num_dirs_mes)), np.diag(cov))

        self.H = np.kron(np.identity(dim), np.eye((num_dirs_mes+1), (num_dirs_gt+1)))

        self.measure_dim = dim*(num_dirs_mes+1)

    def measure_ground_truths(self, gt):
        measurements = []
        measurements_gt = []

        for x in gt.ground_truth:
            noise = np.random.multivariate_normal(np.zeros(self.measure_dim), self.R)
            y = self.measure(x)
            measurements_gt.append(y)
            measurements.append(y + noise)

        return Measurement(measurements_gt=measurements_gt,
                           measurements=measurements,
                           R=self.R,
                           H=self.H,
                           measurement_method=self.measure,
                           sensor="XYZSensor")


    def measure(self, x):
        return self.H @ x 
    
class RBESensor(Sensor):
    def __init__(self, dim=3, num_dirs=1, cov=[10, 0.2, 0.2]):

        self.R = np.diag(cov)

    def measure_ground_truths(self, gt):
        measurements = []
        measurements_gt = []
        
        for x in gt.ground_truth:
            noise = np.random.multivariate_normal(np.zeros(3), self.R)
            y = self.measure(x)
            measurements_gt.append(y)
            measurements.append(y + noise)

        return Measurement(measurements_gt=measurements_gt,
                           measurements=measurements,
                           R=self.R,
                           H=None,
                           measurement_method=self.measure,
                           sensor="RBESensor")

    def measure(self, x):
        return cart_to_rbe(x[0::int(len(x)/3)])
    