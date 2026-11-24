from random import randint,random

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

from benchmark_distr import sample_funnel, log_neal_funnel

mean = 80
sigma = 130
means = [randint(20,100),randint(20,100)]
means = [80,30]



Sigma = np.array([[sigma,0],
         [0,sigma]])
ts = np.linspace(0,100,100)

def gaussian(t,mu,sigma):
    return np.exp(-(t-mu)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))

def gaussian_ND(x,mu,M):
    d = x.shape[1]
    x= np.atleast_2d(x)
    #print("det", np.linalg.det(M),x-mu,x,mu)
    #print("inv", np.linalg.inv(M))
    diff = x-mu
    Minv = np.linalg.inv(M)
    quad = np.einsum('ij,jk,ik->i', diff, Minv, diff)
    norm = np.sqrt((2*np.pi)**d * np.linalg.det(M))
    return np.exp(-0.5 * quad) / norm


#CHANGE THIS TO CHOOSE THE DISTRIBUTION TO SAMPLE
#neal's funnel
is_log = True

def target_distribution(t):
    return gaussian(t,mean,sigma)

def target_distribution_d(t):

    return log_neal_funnel(t)
    return gaussian_ND(t,means,Sigma)

def potential_energy(x):

    return -np.log(target_distribution(x))

def potential_energy_d(x):
    #return -multivariate_normal(means,Sigma).logpdf(x)
    #print("distr is", target_distribution_d(x))
    if is_log:
        return -target_distribution_d(x)
    return -np.log(target_distribution_d(x))

def potential_gradient(x):
    return (x-mean)/sigma**2

def potential_gradient_d(x):
    h=1e-4
    x = np.atleast_2d(x).astype(float)   # (n_chains, d)
    n_chains, d = x.shape
    grad = np.empty((n_chains, d))
    for i in range(d):
        e = np.zeros(d)
        e[i] = h
        grad[:, i] = (potential_energy_d(x + e) - potential_energy_d(x - e)) / (2 * h)
    return grad
    
    
    return (x-means) @ np.linalg.inv(Sigma)

def Hamiltonian(x,p,mass_matrix):
    return potential_energy(x) + 0.5*p**2 /mass_matrix

def Hamiltonian_d(x, p, mass_matrix):
    Minv = np.linalg.inv(mass_matrix)
    return potential_energy_d(x) + 0.5 * np.sum((p @ Minv) * p, axis=1)   # (20,)

def leapfrog(x,p,dt,L,mass_matrix):
    ps = np.zeros((L+1))
    xs = np.zeros((L+1))
    
    for i in range(L):
        p_half = p-0.5 * dt * potential_gradient(x)#np.gradient(potential_energy(xs[i]))
        x = x + dt / mass_matrix * p_half
        p = p_half - 0.5 * dt * potential_gradient(x)
    return x,p

def leapfrog_d(x,p,dt,L,mass_matrix):
    #print("leapfrog x:",x)
    for i in range(L):
        p_half = p-0.5 * dt * potential_gradient_d(x)#np.gradient(potential_energy(xs[i]))
        #print("addition",p_half @ np.linalg.inv(mass_matrix))
        x = x + dt * p_half @ np.linalg.inv(mass_matrix)  
        p = p_half - 0.5 * dt * potential_gradient_d(x)
    return x,p

def alpha(x_0,x_Ldt,p_0,p_Ldt,mass_matrix):

    return min(1,np.exp(-Hamiltonian(x_Ldt,p_Ldt,mass_matrix))/np.exp(-Hamiltonian(x_0,p_0,mass_matrix)))

def alpha_d(x_0, x_Ldt, p_0, p_Ldt, mass_matrix):
    dH = Hamiltonian_d(x_0, p_0, mass_matrix) - Hamiltonian_d(x_Ldt, p_Ldt, mass_matrix)
    return np.minimum(1.0, np.exp(dH))           # (20,)


def chain(x_0,N,dt,L,mass_matrix):

    X = np.zeros((N))
    X[0] = x_0
    for n in range(N-1):
        p_0 = np.random.normal(0,1)
        x_Ldt,p_Ldt = leapfrog(X[n],p_0,dt,L,mass_matrix)
        al = alpha(X[n],x_Ldt,p_0,p_Ldt,mass_matrix)
        if random() < al:
            X[n+1] = x_Ldt
        else:
            X[n+1] = X[n]

    return X

rng = np.random.default_rng()

#hamiltonian
def dchain(x_0,N,dt,L,mass_matrix):
    #needs x_0 in shape (N_chains,d)
    X = np.zeros((N,len(x_0),len(x_0[0])))
    X[0] = x_0
    print(x_0.shape)
    for n in range(N-1):
        p_0 = np.squeeze(np.random.default_rng().multivariate_normal([0,0], mass_matrix, size=len(x_0)))
        #print("p:",p_0)
        x_Ldt,p_Ldt = leapfrog_d(X[n],p_0,dt,L,mass_matrix)
        al = alpha_d(X[n], x_Ldt, p_0, p_Ldt, mass_matrix)
        accept = rng.random(len(al)) < al                # (20,) booleans
        X[n+1] = np.where(accept[:, None], x_Ldt, X[n])
        #print("Xn:",X[n],n)

    return X

#EFFICIENT NO U TURN SAMPLER from paper


d = 2
#GAUSSIAN
"""L = 1000 #leapfrog steps
dt = 0.5 #step size
M = np.array([[10000.0,0],
              [0,10000.0]])
print("the inverse mass matrix is:",np.linalg.inv(M))

p_start = np.random.normal(0,1)
p_start = np.random.default_rng().multivariate_normal([0,0], M, size=1)

#now computes alll chains at once
X0s = [[random()*100,random()*100] for i in range(10)]
Xns = np.array(dchain(np.array(X0s),100,dt,L,M))
plt.plot(Xns[:,0,0],Xns[:,0,1],color="red")


#Grid
x = np.linspace(0,100,100)
y = np.linspace(0,100,100)
X,Y = np.meshgrid(x,y)
pos = np.dstack((X,Y))

Z = multivariate_normal(means,Sigma).pdf(pos)




cf = plt.contourf(X,Y,Z,levels=10,cmap="viridis")
#plt.contour(X,Y,Z,levels=8,colors="white",linewidths = 0.5)
plt.colorbar(cf)
plt.show()

for Xn in Xns:
    plt.plot(Xn[:,0])
    plt.plot(Xn[:,1])
plt.show()
print(np.mean(Xn[:,0]),np.mean(Xn[:,1]))"""

#NEALS FUNNEL
L = 25
dt = 0.2
M = 10*np.eye(2)

X0s = rng.normal(size=(10, 2))
Xns = np.array(dchain(np.array(X0s),100,dt,L,M))
plt.plot(Xns[:,0,0],Xns[:,0,1],color="red")

#plot neals funnel
x = np.linspace(-20, 20, 300)
v = np.linspace(-10, 10, 300)
X, V = np.meshgrid(x, v)          
pos = np.dstack((V, X))            

logp = log_neal_funnel(pos)        

cf = plt.contourf(V, X, np.maximum(logp, -12), levels=20, cmap="viridis")
plt.colorbar(cf, label="log p(v, x)")
plt.xlabel("x")
plt.ylabel("v")
plt.show()

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
print(chains[:,-1])
print(mean,std)

plt.axvline(mean)
plt.plot(ts, target_distribution(ts))
plt.show()"""





"""plt.plot(Xn,target_distribution(Xn))
plt.plot(ts, target_distribution(ts))
plt.legend()
plt.show()"""




