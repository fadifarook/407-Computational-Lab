import numpy as np 
import scipy
import matplotlib.pyplot as plt


"""Central Numerical Differentiation"""

def f(x):
    return np.exp(-x**2)


h_array = np.array([1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10,
                    1e-9, 1e-8, 1e-7, 1e-6, 1e-5,
                    1e-4, 1e-3, 1e-2, 1e-1, 1e0])



def central_difference(func, x, h):
    return (func(x + h/2) - func(x - h/2)) / h
    
def forward_difference(func, x, h):
    return (func(x + h) - func(x)) / h


"""Part A"""

differential_array = np.zeros(len(h_array))
fdifferential_array = np.zeros(len(h_array))
x = 0.5

for i in range(len(h_array)):
    h = h_array[i]

    differential_array[i] = central_difference(f, x, h)
    fdifferential_array[i] = forward_difference(f, x, h)
    print(f"Differential at 1/2 for h = {h} is {differential_array[i]} or \t{fdifferential_array[i]}")


"""Analytical value"""
analyticalValue = (4* 0.5**2 - 2) * np.exp(-0.5**2)
print(f"Analytical Value : {analyticalValue}")

differential_relative_error = np.zeros(len(h_array))
fdifferential_relative_error = np.zeros(len(h_array))

print("\n\nRelative Error")
for i in range(len(h_array)):
    h = h_array[i]
    differential = differential_array[i]
    fdifferential = fdifferential_array[i]

    differential_relative_error[i] = (differential - analyticalValue)/analyticalValue
    fdifferential_relative_error[i] = (fdifferential - analyticalValue)/analyticalValue

    print(f"Relative error for h = {h} is {differential_relative_error[i]} or \t{fdifferential_relative_error[i]}")

    # Best is 1e-5





"""Part D : Plot the previous stuff in log plot"""
# print(differential_relative_error, fdifferential_relative_error)

plt.plot(h_array, np.abs(differential_relative_error), label='Central')
plt.plot(h_array, np.abs(fdifferential_relative_error), label='Forward')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('h')
plt.legend()
plt.show()



"""Part F"""
def g(x):
    # return np.exp(2*x)
    return np.exp(2*x)


def higher_derivative(func, x, m, h):
    if m > 1:
        return (higher_derivative(func, x + h/2, m - 1, h) - higher_derivative(func, x - h/2, m-1, h))/(h)
    else:
        return central_difference(func, x, h)


h= 1e-6
x = 0

print("First 5 derivatives")
for i in range(1, 6):
    print(f"The derivative of order {i} is {higher_derivative(g, x, i, h)}")  # check with real value

