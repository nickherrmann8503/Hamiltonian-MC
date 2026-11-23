import numpy as np


class quantum_harmonic_oscillator():
    def __init__(self, m, k):
        self.m = m
        self.k = k

    def Hamiltonian(self, q, p):
        return 1/2 *(p**2/self.m + self.k*q**2)
    