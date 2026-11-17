import numpy as np
from HamiltonianSampler import chain, target_distribution
from random import randint,random
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

#first implement for 1D
### Gelman-Rubin: choose R<1.1
#Steps:

#1. Run M>1 Chains of length 2N


# 2.Discard the first N draws of each chain, leaving N iterations in the chain
def discard_burnin(N,M,chains):
    for i in range(M):
        chain = chains[i]
        chains[i]=chain[-N:]

    return chains

#calculate variance of each chain
def var_per_chain(M,chains):
    s=np.zeros(M)
    for i in range(M):
        s[i]=np.var(chains[i])

    return s

#mean of parameter per chain
def mean_per_chain(M,chains):
    theta_bar = np.zeros(M)
    for i in range(M):
        theta_bar[i]=np.var(chains[i])
    
    return theta_bar

#calculate mean of each parameter across all chains
def mean_all_chains(theta_bar):
    return np.mean(theta_bar)

#3. Calculate the within and between chain variance.

#within chain variance:
def within_var(M,s):
    return 1/M * np.sum(s[i]**2 for i in range(0,M))

#between chain variance:
def between_var(N,M,theta_bar,mu):
    return N/(M-1)* np.sum((theta_bar[i]-mu)**2 for i in range(0,M))

#4. Calculate the estimated variance of \theta as the weighted sum of between and within chain variance.
def variance(N,W,B):
    return (1-1/N)*W + 1/N*B

#5. Calculate the potential scale reduction factor.
def scale_red_factor(var,W):
    return np.sqrt(var/W)

def Gelman_Rubin(chains):

    M, N = chains.shape

    chains = discard_burnin(N,M,chains)

    s=var_per_chain(M,chains)
    theta_bar=mean_per_chain(M,chains)
    mu=mean_all_chains(theta_bar)

    W = within_var(M,s)
    B = between_var(N,M,theta_bar,mu)
    var = variance(N,W,B)
    R_hat = scale_red_factor(var,W)

    return R_hat

L = 10000 #leapfrog steps
dt = 0.5 #step size

chains = []
for i in range(100):
    x_start = random()*100
    Xn = chain(x_start,30,dt,L,1)
    chains.append(Xn)
    plt.plot(Xn)
plt.show()
chains = np.array(chains)
mean = np.sum(chains[:,-1])/100
std = np.std(chains[:,-1])
print(chains[:,-1])
print(mean,std)

R_hat=Gelman_Rubin(chains)

print('Gelman-Rubin: ', R_hat)