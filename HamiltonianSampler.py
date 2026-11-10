from random import randint,random

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
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
    d = len(x)
    #print("det", np.linalg.det(M),x-mu,x,mu)
    #print("inv", np.linalg.inv(M))
    return np.exp(-0.5*(x-mu).T @ np.linalg.inv(M) @ (x-mu))/np.sqrt((2*np.pi)**d * np.linalg.det(M))

def target_distribution(t):
    return gaussian(t,mean,sigma)

def target_distribution_d(t):
    return gaussian_ND(t,means,Sigma)

def potential_energy(x):

    return -np.log(target_distribution(x))

def potential_energy_d(x):
    #return -multivariate_normal(means,Sigma).logpdf(x)
    #print("distr is", target_distribution_d(x))
    return -np.log(target_distribution_d(x))

def potential_gradient(x):
    return (x-mean)/sigma**2

def potential_gradient_d(x):
    return (x-means) @ np.linalg.inv(Sigma)

def Hamiltonian(x,p,mass_matrix):
    return potential_energy(x) + 0.5*p**2 /mass_matrix

def Hamiltonian_d(x,p,mass_matrix):
    #print("potential is ",potential_energy_d(x))
    return potential_energy_d(x) + 0.5*p.T @ np.linalg.inv(mass_matrix) @ p

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

def alpha_d(x_0,x_Ldt,p_0,p_Ldt,mass_matrix):
    #print(np.exp(-Hamiltonian(x_Ldt,p_Ldt,mass_matrix)))
    #print(np.exp(-Hamiltonian_d(x_Ldt,p_Ldt,mass_matrix)))
    #print(np.exp(-Hamiltonian_d(x_0,p_0,mass_matrix)))
    #print(np.exp(-Hamiltonian_d(x_Ldt,p_Ldt,mass_matrix)) @ np.linalg.inv())
    return min(1,np.exp(-Hamiltonian_d(x_Ldt,p_Ldt,mass_matrix)) / np.exp(-Hamiltonian_d(x_0,p_0,mass_matrix)))


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


def dchain(x_0,N,dt,L,mass_matrix):

    X = np.zeros((N,len(x_0)))
    X[0] = x_0
    for n in range(N-1):
        p_0 = np.squeeze(np.random.default_rng().multivariate_normal([0,0], mass_matrix, size=1))
        #print("p:",p_0)
        x_Ldt,p_Ldt = leapfrog_d(X[n],p_0,dt,L,mass_matrix)
        al = alpha_d(X[n],x_Ldt,p_0,p_Ldt,mass_matrix)
        if random() < al:
            X[n+1] = x_Ldt
        else:
            X[n+1] = X[n]
        #print("Xn:",X[n],n)

    return X
       
d = 2

L = 10000 #leapfrog steps
dt = 0.5 #step size
M = np.array([[10000.0,0],
              [0,10000.0]])
print(np.linalg.inv(M))

p_start = np.random.normal(0,1)
p_start = np.random.default_rng().multivariate_normal([0,0], M, size=1)
print(p_start)


Xns = np.array([dchain(np.array([random()*100,random()*100]),100,dt,L,M) for i in range(20)])
plt.plot(Xns[0,:,0],Xns[0,:,1],color="red")


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




