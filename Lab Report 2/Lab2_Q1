import numpy as np 
import scipy
import matplotlib.pyplot as plt

"""Part A : Define Hermite Polynomial"""

def H(n, x):
    """Takes a non-negative integer n and calculates the nth hermite polynomial at x"""

    if n < 0 or int(n)!=n:
        print("Invalid n value")
    
    if n==0:
        return 1.
    
    elif n==1:
        return 2*x
    
    else:
        return 2*x*H(n-1, x) - 2*(n-1)*H(n-2, x)
    



"""Part B: Harmonic Oscillator Wavefunctions"""


x_array = np.linspace(-4, 4, 100)
n_array = np.array([0, 1, 2, 3])

def wavefunction(n, x):
    """returns the wavefunction equation"""

    temp = H(n, x) * np.exp((-x**2)/2)
    
    return temp / np.sqrt( 2**n * scipy.special.factorial(n) * np.sqrt(np.pi) )



for i in n_array:
    plt.plot(x_array, wavefunction(i, x_array), label=f'n = {i}')

plt.legend()
# plt.show()



"""Part C: Potential"""

# Need to do a change of variables integral here and multiply by 2 for potential


def f_func(n, x):
    """Function to integrate"""
    return x**2 * np.abs(wavefunction(n, x))**2  / 2  # check if this is supposed to be np.abs


def g_func(n, z):
    """After Change of variables"""
    return f_func(n, z/(1-z))/((1-z)**2)


"""Code from last week lab"""

# %load gaussxw
from pylab import *
def gaussxw(N):

    # Initial approximation to roots of the Legendre polynomial
    a = linspace(3,4*N-1,N)/(4*N+2)
    x = cos(pi*a+1/(8*N*N*tan(a)))

    # Find roots using Newton's method
    epsilon = 1e-15  # machine precision is 1e-16
    delta = 1.0
    while delta>epsilon:
        p0 = ones(N,float)
        p1 = copy(x)
        for k in range(1,N):
            p0,p1 = p1,((2*k+1)*x*p1-k*p0)/(k+1)
        dp = (N+1)*(p0-x*p1)/(1-x*x)
        dx = p1/dp
        x -= dx
        delta = max(abs(dx))

    # Calculate the weights
    w = 2*(N+1)*(N+1)/(N*N*(1-x*x)*dp*dp)

    return x,w

def gaussxwab(N,a,b):
    x,w = gaussxw(N)
    return 0.5*(b-a)*x+0.5*(b+a),0.5*(b-a)*w




"""-1 to 0"""
a1 = -1
b1 = 0
N = 100

zp1, wp1 = gaussxwab(N, a1, b1)


"""0 to 1"""
a2 = 0
b2 = 1

zp2, wp2 = gaussxwab(N, a2, b2)


"""Integral"""
n_array = np.arange(1, 11)

s = 0.0
for n in n_array:
    for  i in range(N):
        s += wp1[i] * g_func(n, zp1[i])
        s += wp2[i] * g_func(n, zp2[i])

    print(f"Integral for n = {n} is {s}")

    s = 0.0