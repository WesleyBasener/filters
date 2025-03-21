import numpy as np
from dataclasses import dataclass
 
@dataclass
class GroundTruth:
    
    ground_truth:np.array
    Q:np.array
    F:np.array
    motion_method:str


def CNWP(n:int, dim:int=3, init:np.array=None, cov:np.array=None, dt=1):
    #assert cov.shape == (2,2)
    #assert len(init) == 2*dim

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

    return GroundTruth(ground_truth=gt, Q=Q, F=F, motion_method="CNWP")

def GroundTruthFactory(model:str, n:int, dim:int=3, init:np.array=None, cov:np.array=None, dt=1):

    if model == "CNWP":
        return CNWP(n=n, dim=dim, init=init, cov=cov, dt=dt)

