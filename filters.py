import numpy as np
import scipy
from jax import jacfwd

from utils import cart_to_rbe, rbe_to_cart
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
        y = z - self.H @ self.x_pred

        S = self.H @ self.P_pred @ self.H.T + self.R
        K = self.P_pred @ self.H.T @ np.linalg.inv(S)

        self.x = self.x_pred + K @ y
        self.P = (np.identity(self.P.shape[0]) - K @ self.H) @ self.P_pred

class ExtendedKalmanFilter(Filter):
    def __init__(self, gt, mes, init_x=None, init_P=None):
        self.F = gt.F
        self.Q = gt.Q
        self.R = mes.R
        self.measurement_method = mes.measurement_method
        self.H = jacfwd(mes.measurement_method)
        self.F_jac = jacfwd(lambda x : x @ self.F )

        if init_x is None:
            self.x = gt.ground_truth[0]
        else:
            self.x = init_x

        if init_P is None:
            self.P = np.identity(len(self.x))
        else:
            self.P = init_P

        # self.count = 0
        # self.Q_scale = 10
        # self.e_thresh = 100

    def predict_and_update(self, z, dt = 1):
        self.predict()
        self.update(z)

    def predict(self):
        F_jac = self.F_jac(self.x.astype(float))
        self.x_pred = self.F @ self.x
        self.P_pred = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        H = self.H(self.x_pred)

        y = z - self.measurement_method(self.x_pred)

        S = H @ self.P_pred @ H.T + self.R
        K = self.P_pred @ H.T @ np.linalg.inv(S)

        self.x = self.x_pred + K @ y
        self.P = (np.identity(self.P.shape[0]) - K @ H) @ self.P_pred
   
        # e = y.T @ np.linalg.inv(S) @ y
        # if e > self.e_thresh:
        #     self.count += 1
        #     self.Q *= self.Q_scale
        # elif self.count > 0:
        #     self.count -= 1
        #     self.Q /= self.Q_scale

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
    def __init__(self, gt, mes, num_part:int=1000, nef_thresh:float=0.9, init_x:np.array=None, init_P:np.array=None):
        self.F = gt.F
        self.Q = gt.Q
        self.R = mes.R
        self.measurement_method = mes.measurement_method
        self.num_part = num_part
        self.nef_thresh = nef_thresh
        if init_x is None:
            self.x = gt.ground_truth[0]
        else:
            self.x = init_x

        #generate the initial particles
        self.particles = np.full(shape=(self.num_part, len(self.x)), fill_value=self.x)
        #self.particles = np.random.multivariate_normal(self.x, self.Q, size=self.num_part)

        self.weights = np.full(shape=self.num_part, fill_value=1/self.num_part)

    def predict_and_update(self, z):
        self.predict()
        self.update(z)

    def predict(self):
        for i in range(self.num_part):
            self.particles[i] = np.random.multivariate_normal(self.particles[i], self.Q)
        self.particles = self.particles @ self.F.T

    def update(self, z):
        Z = self.measure(self.particles)

        for i in range(self.num_part):
            self.weights[i] = self.weights[i] * scipy.stats.multivariate_normal(Z[i], self.R).pdf(z)

        self.weights /= np.sum(self.weights)

        self.x = self.weights @ self.particles

        if self.get_nef() < self.nef_thresh:
            self.resample()

    def measure(self, X):
        Z = np.zeros((X.shape[0], self.R.shape[0]))
        for i in range(X.shape[0]):
            Z[i] = self.measurement_method(X[i])
        return Z
    
    def get_nef(self):
        return 1/(self.num_part * np.sum(self.weights**2))
    
    def resample(self):
        P = self.get_covariance()
        self.particles = np.random.multivariate_normal(self.x, P, size=self.num_part)
        self.weights[:] = 1/self.num_part

    def get_covariance(self):
        covar = np.zeros(self.Q.shape)
        for i in range(self.num_part):
            std = self.particles[i] - self.x
            covar += self.weights[i] * np.outer(std, std)
        return covar