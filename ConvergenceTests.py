import numpy as np
from numpy import linalg
from HamiltonianSampler import dchain, chain
from random import randint,random
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

#first implement for 1D
### Gelman-Rubin: aim for Rhat<1.1 or (for rank-normalized split Rhat) Rhat<1.01

#calculate variance of each chain
def var_per_chain(chains):
    return np.std(chains, axis=0, ddof=1)

#mean of parameter per chain
def mean_per_chain(chains):    
    return np.mean(chains, axis=0)

#calculate mean of each parameter across all chains
def mean_all_chains(theta_bar):
    return np.mean(theta_bar, axis=0)

#3. Calculate the within and between chain variance.

#within chain variance:
def within_var(m,s):
    return 1/m * np.sum(s**2, axis=0)

#between chain variance:
def between_var(n,m,theta_bar,mu):
    return n/(m-1)* np.sum((theta_bar-mu)**2, axis=0)

#4. Calculate the estimated variance of \theta as the weighted sum of between and within chain variance.
def variance(n,W,B):
    return (1-1/n)*W + 1/n*B

#5. Calculate the potential scale reduction factor.
def scale_red_factor(var,W):
    return np.sqrt(var/W)

def Gelman_Rubin_single(chains):

    #account for 1D case
    if chains.ndim == 2:
        chains = chains[:, :, None]

    l, m, dim = chains.shape
    n = l // 2

    #discard burn in
    chains = chains[-n:, :]

    s=var_per_chain(chains)
    theta_bar=mean_per_chain(chains)
    mu=mean_all_chains(theta_bar)

    wit = within_var(m,s)
    bet = between_var(n,m,theta_bar,mu)
    var = variance(n,wit,bet)
    R_hat = scale_red_factor(var,wit)

    return R_hat

def Gelman_Rubin(chains):
    Rhat_all=[]
    l, m, dim = chains.shape

    for i in range(2,l):
        chain = chains[:i, :].copy()
        Rhat_all.append(Gelman_Rubin_single(chain))

    Rhat_all = np.array(Rhat_all)
    
    return Rhat_all


#multivariate Effective sample size (mESS)

#Σ is the asymptotic covariance matrix in the Markov chain CLT 
#(different from Λ since samples are correlated.

#different source: SIGMA is an estimate of the Monte 
#Carlo covariance matrix for the Markov chain (here obtained by batch estimation).

#sample covariance matrix
def cov(chains, b, a):
    flat = chains[:a*b, :].reshape(-1, chains.shape[-1])              #chains[:, :a*b] so it's cleaner, since I also used that for sigma
    return np.atleast_2d(np.cov(flat, rowvar=False))                  #rowvar=False st. columns are variables, rows are observations

#batch estimation
#rule of thumb: batch size of sqrt(N)
#let T={.1, .2, .3, .4, .5}, and take mESS^*=\min _{t \in T, b_n=n^t} mESS(b_n)
def batch_size(n, t):
    b = max(1, int(n**t))
    return b, n//b

#\bar{Y}_k:=b_n^{-1} \sum_{t=1}^{b_n} Y_{k b_n+t} .
def batch_mean(chains, b, a, m, dim):
    return chains[:a*b, :].reshape(a, b, m, dim).mean(axis=1)

#estimate of the monte carlo covariance matrix for the Markov Chain
#\Sigma_n=\frac{b_n}{a_n-1} \sum_{k=0}^{a_n-1}\left(\bar{Y}_k-\theta_n\right)\left(\bar{Y}_k-\theta_n\right)^T .
def MCcov(chains, b, a,m,dim):
    Ybar=batch_mean(chains,b,a,m,dim).reshape(m*a, dim)
    mu = chains[:a*b, :].reshape(-1, dim).mean(axis=0)
    Rm=Ybar-mu
    return b/(m*a-1) * Rm.T@Rm

def mESS_single(chains):                      #here I pooled batches over different chains-> if different chains have different means, this will lower mESS
    #account for 1D case                      #maybe it would be better to find mESS for each chain separately and take mean of all chains    
    if chains.ndim == 2:
        chains = chains[:, :, None]
    
    l, m, dim = chains.shape
    n = l // 2
    
    #discard burn in
    chains = chains[-n:, :]

    mess = []
    for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
        b, a = batch_size(n,t)

        if m*a <= dim:          
            continue

        lam=cov(chains, b, a)
        
        #use slogdet to avoid over-/underflow
        _, logdet_lam = np.linalg.slogdet(lam)

        sig=MCcov(chains, b, a, m, dim)
        _, logdet_sig = np.linalg.slogdet(sig)

        mess.append(n*m*np.exp((logdet_lam-logdet_sig)/dim))

    mess=np.array(mess)

    return np.min(mess)

def mESS_single_t(chains):                      
    #account for 1D case                        
    if chains.ndim == 2:
        chains = chains[:, :, None]
    
    l, m, dim = chains.shape
    n = l // 2
    
    #discard burn in
    chains = chains[-n:, :]

    mess = []
    for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
        b, a = batch_size(n,t)

        if m*a <= dim:          
            continue

        lam=cov(chains, b, a)
        
        #use slogdet to avoid over-/underflow
        _, logdet_lam = np.linalg.slogdet(lam)

        sig=MCcov(chains, b, a, m, dim)
        _, logdet_sig = np.linalg.slogdet(sig)

        mess.append(n*m*np.exp((logdet_lam-logdet_sig)/dim))

    return np.array(mess)

def mESS(chains):
    mess_all=[]
    l, m, dim = chains.shape

    for i in range(2,l):
        chain = chains[:i, :].copy()
        mess_all.append(mESS_single(chain))

    mess_all = np.array(mess_all)
    
    return mess_all

def mESS_t(chains):
    mess_all=[]
    l, m, dim = chains.shape

    for i in range(2,l):
        chain = chains[:i, :].copy()
        mess_all.append(mESS_single_t(chain))

    mess_all = np.array(mess_all)
    
    return mess_all

L = 100 #leapfrog steps
dt = 0.5 #step size
M = np.array([[10000.0,0],
              [0,10000.0]])
print("the inverse mass matrix is:",np.linalg.inv(M))

p_start = np.random.normal(0,1)
p_start = np.random.default_rng().multivariate_normal([0,0], M, size=1)

#now computes alll chains at once
X0s = [[random()*100,random()*100] for i in range(1000)]
Xns = np.array(dchain(np.array(X0s),100,dt,L,M))

#print('chains shape: ', Xns.shape)

R_hat=Gelman_Rubin(Xns)

print('Gelman-Rubin: ', R_hat)
#mess=(mESS(Xns))
#print('multivariate effective sample size: ', mess)

#plot mESS
def plot_mESS(mess, l, m, log=False):
    lengths = np.arange(2, l)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(lengths, mess, "o-", lw=2, ms=4, label="mESS")
    ax.plot(lengths, m*(lengths//2), "k--", lw=0.8, alpha=0.5, label="mESS = n")  # i.i.d. upper bound
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("chain length")
    ax.set_ylabel("mESS")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax

#plot Efficiency
def plot_eff(mess, l, m, log=False):
    lengths = np.arange(2, l)
    ones=np.ones(l-2)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(lengths, mess/(m*(lengths//2)), "o-", lw=2, ms=4, label="mESS/n")
    ax.plot(lengths, ones, "k--", lw=0.8, alpha=0.5, label="mESS = n")  # i.i.d. upper bound
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("chain length")
    ax.set_ylabel("mESS/n")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax

#l, m, dim = Xns.shape
#fig, ax = plot_mESS(mess,l,m)
#fig, ax = plot_eff(mess,l,m)
#plt.show()

# issue: sawtooth structure in efficiency plot -> plot efficiency for different t's/batch sizes separately
# result: probably some bias in logdet estimate of sigma, because estimates rise for larger t
#mess_t = mESS_t(Xns)
#for i,t in zip(range(5),[0.1,0.2,0.3,0.4,0.5]):
#    fig, ax = plot_eff(mess_t[:,i],l,m)
#    ax.set_title('$t={t}$')
#    plt.show()

def NUTS_divergence_diagnostic(H, T, d_t):
    """
    parameters:
    H: Hamiltonian energy per chain
    d_t: divergence flag per chain
    T: total number of NUTS iterations
    
    returns:
    diagnostic: low values (approx. <0.3) indicate inefficient exploration of energy distribution
    divergence_rate: fraction of divergent NUTS iterations"""
    diagnostic = 0
    for t in range(1, len(H)):
        diagnostic += (H[t] - H[t-1])**2 

    diagnostic /= np.sum((H - mean_per_chain(H))**2)

    divergence_rate = 1/T * np.sum(d_t) 

    return diagnostic, divergence_rate

def NUTS_divergence_diagnostic_verdict(H, T, d_t):
    diagnostic, divergence_rate = NUTS_divergence_diagnostic(H, T, d_t)
    if diagnostic < 0.3 and divergence_rate > 0.01:
        print("VERDICT: chain is unreliable - reparameterize or raise adapt_delta.")
    else:
        print("VERDICT: chain is healthy.")


#sanity test: works -> mESS estimator should be working
#iid = np.random.randn(200, 1000, 2)
#print(mESS_single(iid), (200//2)*1000)

moved = np.any(Xns[1:] != Xns[:-1], axis=2)   # (n_steps-1, m)
print(1 - moved.mean())

print(Xns[-1].mean(axis=0), Xns[-1].std(axis=0))