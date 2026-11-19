import numpy as np
import scipy as sp
from numpy.random import multivariate_normal
import matplotlib.pyplot as plt

# Multivariate Gaussian (n-dimensional) with variable correlation

def gaussian(x,mu,M):
    d = x.size
    return np.exp(-0.5*(x-mu).T @ np.linalg.inv(M) @ (x-mu))/np.sqrt((2*np.pi)**d * np.linalg.det(M))


def log_gaussian(x,mu,M):
    d = x.size
    return -0.5*(x-mu).T @ np.linalg.inv(M) @ (x-mu) - 0.5 * np.log((2*np.pi)**d * np.linalg.det(M))

def grad_log_gaussian(x,mu,M):
    return - np.linalg.inv(M) @ (x-mu)

# Neal's Funnel

def sample_funnel(n_samples, dim=9, sigma_v=3.0, sigma_x=1.0, rng=None):
    rng = np.random.default_rng(rng)
    v = rng.normal(0, sigma_v, size=n_samples)
    x = sigma_x * rng.normal(size=(n_samples, dim)) * np.exp(v / 2)[:, None]
    return v, x

def log_neal_funnel(theta, sigma_v=3.0, sigma_x=1.0):
    v, x = theta[0], theta[1:]
    n = x.size
    return (-v**2 / (2 * sigma_v**2) - 0.5 * n * v - 0.5 * np.sum(x**2) * np.exp(-v) / sigma_x**2)


def grad_log_neal_funnel(theta, sigma_v=3.0, sigma_x=1.0):
    v, x = theta[0], theta[1:]
    n = x.size
    ev = np.exp(-v) / sigma_x**2
    dv = -v / sigma_v**2 - 0.5 * n + 0.5 * np.sum(x**2) * ev
    dx = -x * ev
    return np.concatenate([[dv], dx])


linspace = np.linspace(-10,10)
y = np.empty(linspace.size)
for j in range(5):
    for i,val in enumerate(linspace):
        q = np.array([j,val])
        y[i] = log_neal_funnel(q)
    plt.plot(linspace,np.exp(y))


plt.show()
"""
print(v.size,x.size)
for i in range(9):
    plt.scatter(x[:,i],v)

plt.show()
"""

# Multiple Gaussian Overlap (n-dimesional)


# maybe a banana?

