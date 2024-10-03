import numpy as np 
import scipy
import matplotlib.pyplot as plt


"""Central Numerical Differentiation"""

def f(x):
    return np.exp(-x**2)


h_array = np.array([1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10,
                    1e-9, 1e-8, 1e-7, 1e-6, 1e-5,
                    1e-4, 1e-3, 1e-2, 1e-1, 1e0])


# Using the one given in the question paper

def central_difference(x_array, f_array):
    
    deltax = x_array[1] - x_array[0]

    dfdx = np.zeros(len(x_array))

    # forward for start, backward for end
    dfdx[0] = (f_array[1] - f_array[0])/deltax
    dfdx[-1] = (f_array[-1] - f_array[-2])/deltax

    # for the rest
    for i in range(1, len(x_array)-1):
        dfdx[i] = (f_array[i+1] - f_array[i-1])/(2 * deltax)
    
    return dfdx



"""Part A"""

differential_array = np.zeros(len(h_array))
for i in range(len(h_array)):
    h = h_array[i]
    x_array = np.array([0.5-h, 0.5, 0.5+h])
    f_array = f(x_array)

    differential_array[i] = central_difference(x_array, f_array)[1]
    print(f"Differential at 1/2 for h = {h} is {differential_array[i]}")
    


"""Analytical value"""
analyticalValue = (4* 0.5**2 - 2) * np.exp(-0.5**2)
print(f"Analytical Value : {analyticalValue}")

print("\n\nRelative Error")
for i in range(len(h_array)):
    h = h_array[i]
    differential = differential_array[i]

    print(f"Relative error for h = {h} is {(differential - analyticalValue)/analyticalValue}")

    # Best is 1e-5










"""Part F"""
def g(x):
    return np.exp(2*x)


def higher_derivative(func, x, m, h):
    if m > 1:
        return (higher_derivative(f, x + h/2, m - 1, h) - higher_derivative(f, x - h/2, m-1, h))/(h)
    else:
        return (func(x + h/2) - func(x - h/2))/(h)


h= 1e-6
x = 0

print("First 5 derivatives")
for i in range(1, 6):
    print(f"The derivative of order {i} is {higher_derivative(g, x, i, h)}")

