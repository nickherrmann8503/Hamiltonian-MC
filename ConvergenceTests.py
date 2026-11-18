import numpy as np
from HamiltonianSampler import dchain, chain
from random import randint,random
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

#first implement for 1D
### Gelman-Rubin: aim for Rhat<1.1 or (for rank-normalized split Rhat) Rhat<1.01

#calculate variance of each chain
def var_per_chain(chains):
    return np.std(chains, axis=1, ddof=1)

#mean of parameter per chain
def mean_per_chain(chains):    
    return np.mean(chains, axis=1)

#calculate mean of each parameter across all chains
def mean_all_chains(theta_bar):
    return np.mean(theta_bar, axis=0)

#3. Calculate the within and between chain variance.

#within chain variance:
def within_var(M,s):
    return 1/M * np.sum(s**2, axis=0)

#between chain variance:
def between_var(N,M,theta_bar,mu):
    return N/(M-1)* np.sum((theta_bar-mu)**2, axis=0)

#4. Calculate the estimated variance of \theta as the weighted sum of between and within chain variance.
def variance(N,W,B):
    return (1-1/N)*W + 1/N*B

#5. Calculate the potential scale reduction factor.
def scale_red_factor(var,W):
    return np.sqrt(var/W)

def Gelman_Rubin(chains):

    #account for 1D case
    if chains.ndim == 2:
        chains = chains[:, :, None]

    M, L, D = chains.shape
    N = L // 2

    #discard burn in
    chains = chains[:, -N:]

    s=var_per_chain(chains)
    theta_bar=mean_per_chain(chains)
    mu=mean_all_chains(theta_bar)

    W = within_var(M,s)
    B = between_var(N,M,theta_bar,mu)
    var = variance(N,W,B)
    R_hat = scale_red_factor(var,W)

    return R_hat


#multivariate Effective sample size (mESS)

#sample covariance matrix
def cov(chains):
    flat = chains.reshape(-1, chains.shape[-1])
    return np.cov(flat, rowvar=False)

#batch estimation

#estimate of the monte carlo covariance matrix for the Markov Chain
def MCcov(chains):
    return chains

def mESS(chains):
    #account for 1D case
    if chains.ndim == 2:
        chains = chains[:, :, None]
    
    M, L, D = chains.shape
    N = L // 2
    
    #discard burn in
    chains = chains[:, -N:]

    lam=cov(chains)
    sig=MCcov(chains)

    mESS=N*lam**(1/D)/sig**(1/D)

    return mESS

L = 100 #leapfrog steps
dt = 0.5 #step size

d = 2

M = np.array([[10000.0,0],
              [0,10000.0]])
print(np.linalg.inv(M))

p_start = np.random.normal(0,1)
p_start = np.random.default_rng().multivariate_normal([0,0], M, size=1)
print(p_start)

Xns = np.array([dchain(np.array([random()*100,random()*100]),100,dt,L,M) for i in range(20)])

#print('chains shape: ', Xns.shape)

#1D
"""chains = []
for i in range(100):
    x_start = random()*100
    Xn = chain(x_start,30,dt,L,1)
    chains.append(Xn)
    plt.plot(Xn)
plt.show()
chains = np.array(chains)
mean = np.sum(chains[:,-1])/100
std = np.std(chains[:,-1])
#print(chains[:,-1])
#print(mean,std)
"""

#R_hat=Gelman_Rubin(Xns)

#print('Gelman-Rubin: ', R_hat)