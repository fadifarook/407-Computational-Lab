import numpy as np 
import scipy
import matplotlib.pyplot as plt
from gaussxw import gaussianQuadrature, gaussxw

"""Code that calculates the wavefunction of a quantum harmonic oscillator. Plots the wavefunction
and calculates the associated potential.

Imports the function gaussianQuadrature from gaussxw module I wrote. It calculates the integral and returns
xpoints, weightedpoints and integral in a tuple of that order. It takes the 0 to 1 x points and weights calculated
from the gaussxw function"""


"""Part A : Define Hermite Polynomial"""

def H(x, n):
    """Takes a non-negative integer n and calculates the nth hermite polynomial at x"""

    if n < 0 or int(n)!=n:
        print("Invalid n value")  # Ensure that n is a non-negative integer
    
    if n==0:
        return 1.  # Base case for n=0
    
    elif n==1:
        return 2*x  # Base case for n=1
    
    else:
        return 2*x*H(x, n-1) - 2*(n-1)*H(x, n-2)  # Recursive definition of Hermite polynomials
    



"""Part B: Harmonic Oscillator Wavefunctions"""

def wavefunction(x, n):
    """returns the wavefunction equation"""

    temp = H(x, n) * np.exp((-x**2)/2)
    
    return temp / np.sqrt( 2**n * scipy.special.factorial(n) * np.sqrt(np.pi) )



x_array = np.linspace(-4, 4, 100)  # Define a range of x values for plotting
n_array = np.array([0, 1, 2, 3])  # Define the energy levels from 0 to 3


for i in n_array:
    plt.plot(x_array, wavefunction(x_array, i), label=f'n = {i}')  # Plot wavefunction for each n

plt.xlabel('x')
plt.ylabel('Energy')
plt.legend()
plt.savefig("Lab2Q1B.png")



"""Part C: Calculate Potential"""

# Need to do a change of variables integral here and divide by 2 for potential


def f_func(x, n):
    """Function to integrate in original x variable"""
    return x**2 * np.abs(wavefunction(x, n))**2

def g_func(z, n):
    """After change of variables: x = tan(z)"""
    return f_func(np.tan(z), n) / np.cos(z)**2  # Adjust the function using change of variables from x to z



"""Integral bounds"""
a, b = -np.pi/2, np.pi/2
N = 100  # Sample points for Gaussian Quadrature

n_array = np.arange(0, 11)  # Energy Levels

# Generate the x and weight arrays for Gaussian quadrature once
x_initial, w_initial = gaussxw(N)

for n in n_array:

    integral_result = gaussianQuadrature(g_func, N, a, b, x_initial, w_initial, n)[2]  # Integral calculated for each n

    # Note that this makes the same calculation converting from 0 to 1 -> a to b, but it is a minimal
    # calculation that is fast.

    # Calculate potential
    potential = integral_result / 2
    
    print(f"n = {n}: \t Integral {integral_result} \t Potential: {potential}")