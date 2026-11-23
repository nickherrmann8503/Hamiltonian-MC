import numpy as np


def HMC(U, grad_U, epsilon, L, current_q):
    """
    U: target distribution
    grad_U: 
    epsilon: stepsize
    L: number of steps
    current_q: current chain link?
    """
    q = current_q
    p = np.random.normal(loc=0, scale=1, size=q.shape)
    current_p = p

    p -= epsilon*grad_U(q)/2 #half step in p

    for i in range(L):
        q += epsilon*p #full step in q
        if i == L-1:
            break
        p -= epsilon*grad_U(q) #full step in p

    p -= epsilon*grad_U(q)/2 #half step in p

    p = -p

    current_U = U(current_q)
    current_K = sum(current_p^2) / 2
    proposed_U = U(q)
    proposed_K = sum(p^2) / 2

    if np.random.uniform() < np.exp(current_U - proposed_U + current_K - proposed_K):
        return q
    else:
        return current_q
    
