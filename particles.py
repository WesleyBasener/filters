import numpy as np
from numpy import sin, cos
from dataclasses import dataclass
from typing import Callable

@dataclass
class GroundTruth:
    
    ground_truth:np.array
    dim:int
    num_dirs:int
    Q:np.array
    F:np.array
    linear:bool
    motion_method:Callable
    motion_model:str


def CWNAM(n:int, dim:int=3, init:np.array=None, cov:np.array=None, dt=1):
    """
    Continuous White Noise Acceleration Model
    Args:
        n (int): number of sample points
        dim (int, optional): Dimesnion of movement. Defaults to 3.
        init (np.array, optional): initial location. Defaults to None.
        cov (np.array, optional): motion covariance matrix for each dimension. Defaults to None.
        dt (int, optional): time change between steps. Defaults to 1.

    Returns:
        GroundTruth dataclass object 
    """

    if cov is None:
        cov = np.array([[0.1, 0],
                        [0,   1]])

    Q = np.kron(np.identity(dim), cov)    
    F = np.kron(np.identity(dim),[[1, dt], [0, 1]])

    gt = []

    if init is None:
        gt.append(np.zeros(2*dim))
    else:
        gt.append(init)

    for i in range(n):
        #create the noise
        noise = np.random.multivariate_normal(np.zeros(2*dim), Q)

        #propogate last 
        next = F @ gt[i] + noise
        gt.append(next)

    return GroundTruth(ground_truth=gt,
                        dim=dim, 
                        num_dirs=1, 
                        Q=Q, 
                        F=F,
                        linear=True, 
                        motion_method=None,
                        motion_model="CNWP")

def CWPAN(n:int, dim:int=3, init:np.array=None, cov:np.array=None, dt=1, psd:float=None):
    """
    Continuous Wiener Process Acceleration Model
    Args:
        n (int): number of sample points
        dim (int, optional): Dimesnion of movement. Defaults to 3.
        init (np.array, optional): initial location. Defaults to None.
        cov (np.array, optional): motion covariance matrix for each dimension. Defaults to None.
        dt (int, optional): time change between steps. Defaults to 1.

    Returns:
        GroundTruth dataclass object 
    """

    if psd is None:
        psd = 1

    if cov is None:
        cov = psd*np.array([[(dt**5)/20, (dt**4)/8, (dt**3)/6],
                            [(dt**4)/8,  (dt**3)/3, (dt**2)/2],
                            [(dt**3)/6,  (dt**2)/2,  dt]])

    Q = np.kron(np.identity(dim), cov)    
    F = np.kron(np.identity(dim),[[1, dt, 0.5*dt**2], 
                                  [0, 1,  dt], 
                                  [0, 0,  1]])

    gt = []

    if init is None:
        gt.append(np.zeros(3*dim))
    else:
        gt.append(init)

    for i in range(n):
        #create the noise
        noise = np.random.multivariate_normal(np.zeros(3*dim), Q)

        #propogate last 
        next = F @ gt[i] + noise
        gt.append(next)

    return GroundTruth(ground_truth=gt, 
                       dim=dim, 
                       num_dirs=2, 
                       Q=Q, 
                       F=F,
                       linear=True,
                       motion_method=None,
                       motion_model="CNWP")



def GroundTruthFactory(model:str, n:int, dim:int=3, psd:float=1.0, init:np.array=None, cov:np.array=None, dt=1):

    if model == "CWNAM":
        return CWNAM(n=n, dim=dim, init=init, cov=cov, dt=dt)
    if model == "CWPAN":
        return CWPAN(n=n, dim=dim, init=init, cov=cov, dt=dt, psd=psd)
