from random import randint,random

import numpy as np
import matplotlib.pyplot as plt
mean = 80
sigma = 10

ts = np.linspace(0,100,100)

def gaussian(t,mu,sigma):
    return np.exp(-(t-mu)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))

def gaussian_ND(x,mu,M):
    d = len(x)
    return np.exp(-0.5*(x-mu).T @ np.linalg.inv(M) @ (x-mu))/np.sqrt((2*np.pi)**d * np.linalg.det(M))

def target_distribution(t):
    return gaussian(t,mean,sigma)

def potential_energy(x):

    return -np.log(target_distribution(x))

def potential_gradient(x):
    return (x-mean)/sigma**2

def Hamiltonian(x,p,mass_matrix):
    return potential_energy(x) + 0.5*p**2 /mass_matrix

def leapfrog(x,p,dt,L,mass_matrix):
    ps = np.zeros((L+1))
    xs = np.zeros((L+1))
    
    for i in range(L):
        p_half = p-0.5 * dt * potential_gradient(x)#np.gradient(potential_energy(xs[i]))
        x = x + dt / mass_matrix * p_half
        p = p_half - 0.5 * dt * potential_gradient(x)
    return x,p


def alpha(x_0,x_Ldt,p_0,p_Ldt,mass_matrix):

    return min(1,np.exp(-Hamiltonian(x_Ldt,p_Ldt,mass_matrix))/np.exp(-Hamiltonian(x_0,p_0,mass_matrix)))


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
        


L = 10000 #leapfrog steps
dt = 2 #step size


p_start = np.random.normal(0,1)

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

plt.axvline(mean)
plt.plot(ts, target_distribution(ts))
plt.show()



"""plt.plot(Xn,target_distribution(Xn))
plt.plot(ts, target_distribution(ts))
plt.legend()
plt.show()"""




