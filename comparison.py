# import HMC_isolated
# from benchmark_distr import gaussian
import jax.random
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import emcee
import scipy as sp
import time
from scipy.stats import norm, multivariate_normal

# Benchmark 1: Dimensionality via multivariate Gaussian
dimensions = [2,10,100,500]
sigma = 1
mu = 0
x = np.linspace(-5,5)

rng = np.random.default_rng(42)


def log_prob_gauss(x):
    return -0.5 * np.sum(x**2)


mean_abs_err = np.empty((len(dimensions)))
var_abs_err = np.zeros_like(mean_abs_err)

total_evals = 200000

fig, axes = plt.subplots(2,2, squeeze=False)
axes = axes.ravel()
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

for i,dim in enumerate(dimensions):
    # settings
    ndim = dim
    nwalkers = 2 * ndim + 10
    nsteps = total_evals // nwalkers
    burnin = nsteps // 5

    # initialise x0 and sampler
    x0 = rng.normal(loc=mu,scale=sigma,size=(nwalkers,ndim))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_gauss)

    # run emcee
    start = time.perf_counter()
    sampler.run_mcmc(
        x0,
        nsteps,
        progress=True
    )
    runtime = time.perf_counter() - start
    chain = sampler.get_chain(discard=burnin, flat=True)

    # 1st dimension for plot
    x_samples = chain[:,0]
    y_samples = chain[:,1]

    ax = axes[i]
    ax.scatter(x_samples[:300],y_samples[:300],marker='.',label=f'emcee {dim}d, runtime:{runtime:.2f}',color=colors[i])
    ax.plot(x_samples[:300],y_samples[:300],alpha=0.5,color='grey')
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    ax.legend()
    ax.grid()

    # plt.hist(x_samples,bins=60,density=True,alpha=0.5,label=f'emcee {dim}d, runtime:{runtime:.2f}s')
    """
    # estimated mean and variance 
    estimated_means = np.mean(chain, axis=0)
    estimated_vars = np.var(chain, axis=0)

    dim_arange = np.arange(1,dim+1,dtype=int)
    axes[i].scatter(dim_arange,estimated_means,marker='.',color=colors[i],label='emcee samples estimated mean')
    axes[i].set_ylim(-0.15,0.15)
    axes[i].legend()

    # absolute mean and variance errors
    mean_error = np.mean(np.abs(estimated_means))
    variance_error = np.mean(np.abs(estimated_vars - 1))

    mean_abs_err[i] = mean_error
    var_abs_err[i] = variance_error
    """
#plt.plot(x,norm.pdf(x,mu,sigma))
plt.legend()
plt.show()

"""
plt.plot(np.array(dimensions),mean_abs_err,marker='o',ls='--',label='mean absolute error')
plt.plot(np.array(dimensions),var_abs_err,marker='o',ls='--',label='variance absolute error')
plt.legend()
plt.show()
"""






