import numpy as np 
import scipy
import matplotlib.pyplot as plt
from gaussxw import gaussxwab, gaussianQuadrature

"""Part A : Define Hermite Polynomial"""

def H(x, n):
    """Takes a non-negative integer n and calculates the nth hermite polynomial at x"""

    if n < 0 or int(n)!=n:
        print("Invalid n value")
    
    if n==0:
        return 1.
    
    elif n==1:
        return 2*x
    
    else:
        return 2*x*H(x, n-1) - 2*(n-1)*H(x, n-2)
    



"""Part B: Harmonic Oscillator Wavefunctions"""


x_array = np.linspace(-4, 4, 100)
n_array = np.array([0, 1, 2, 3])

def wavefunction(x, n):
    """returns the wavefunction equation"""

    temp = H(x, n) * np.exp((-x**2)/2)
    
    return temp / np.sqrt( 2**n * scipy.special.factorial(n) * np.sqrt(np.pi) )



for i in n_array:
    plt.plot(x_array, wavefunction(x_array, i), label=f'n = {i}')

plt.legend()
plt.savefig("Lab2Q1B")



"""Part C: Potential"""

# Need to do a change of variables integral here and multiply by 2 for potential


def f_func(x, n):
    """Function to integrate"""
    return x**2 * np.abs(wavefunction(x, n))**2  / 2  # check if this is supposed to be np.abs


def g_func(z, n):
    """After Change of variables"""
    return f_func((z/(1-z))/((1-z)**2), n)


"""-1 to 0"""
a1 = -1
b1 = 0
N = 100


"""0 to 1"""
a2 = 0
b2 = 1


"""Integral"""
n_array = np.arange(0, 11)

s = 0.0
for n in n_array:

    s += gaussianQuadrature(g_func, N, a1, b1, n)[2]
    s += gaussianQuadrature(g_func, N, a2, b2, n)[2]

    print(f"Integral for n = {n} is {s}")

    s = 0.0