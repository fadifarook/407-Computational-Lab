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

plt.xlabel('x')
plt.ylabel('Energy')
plt.legend()
plt.savefig("Lab2Q1B")



"""Part C: Potential"""

# Need to do a change of variables integral here and divide by 2 for potential


def f_func(x, n):
    """Function to integrate in original x variable"""
    return x**2 * np.abs(wavefunction(x, n))**2

def g_func(z, n):
    """After change of variables: z = x / (1 + x)"""
    return f_func(np.tan(z), n) / np.cos(z)**2



"""Integral bounds"""
a, b = -np.pi/2, np.pi/2
N = 100  # Total points for Gaussian Quadrature

n_array = np.arange(0, 11)  # Quantum number range

for n in n_array:

    integral_result = gaussianQuadrature(g_func, N, a, b, n)[2]

    # Calculate potential
    potential = np.sqrt(integral_result) / 2
    
    print(f"Integral for n = {n} is {integral_result} \t Potential: {potential}")







### Test

def test(x, n):
    return np.exp(-x**2)


def g_func(z, n):
    """After change of variables: z = x / (1 + x)"""
    return test(np.tan(z), n) / np.cos(z)**2

for n in n_array:
    integral = gaussianQuadrature(g_func, 100, -np.pi/2, np.pi/2, n)[2]
    print(f"Test integral result: {integral} == {np.sqrt(np.pi)}")