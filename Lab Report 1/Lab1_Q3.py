import numpy as np
import matplotlib.pyplot as plt
import time


"""This questions asks to integrate 4/1+x^2 from 0 to 1. I compare the real value pi
to calculations using trapezoidal and simpsons rule of various number of slices.
I also practically estimate the error of the trapezoidal rule"""


def f(x):
    "Given equation to integrate"
    return 4 / (1 + x**2)  # answer is Pi


def trapz(func: callable, start, end, N):
    """Trapezoidal integration function"""
    h = (end - start) / N  # size of slice
    s = (func(end) + func(start)) / 2

    for i in range(1, N):  # dont want i == 0
        s += func(start + i * h)

    return s * h


def simpson(func, start, end, N):
    "Simpson integration function"
    h = (end - start) / N  # size of slice
    s = func(start) + func(end)

    for i in range(1, N):

        if i % 2 == 0:
            s += 2 * func(start + i * h)

        else:
            s += 4 * func(start + i * h)

    return s * h / 3


# bounds of integration
a = 0
b = 1

print("Part B")

Ntrapz = 4
Nsimpson = 4

trueValue = np.pi
trapzIntegral = trapz(f, a, b, Ntrapz)
simpsonIntegral = simpson(f, a, b, Nsimpson)

print(
    "Trapezoid rule integral: ",
    trapzIntegral,
    "\t Percentage Difference: ",
    abs(trapzIntegral - trueValue) / trueValue,
    "%",
)
print(
    "Simpson rule integral: ",
    simpsonIntegral,
    "\t Percentage Difference: ",
    abs(simpsonIntegral - trueValue) / trueValue,
    "%",
)


print("\n\nPart C")

# After trial and error, chose these values
Ntrapz = 4096  # number of slices
Nsimpson = 16  # number of slices


start = time.time()
trapzIntegral = trapz(f, a, b, Ntrapz)
checkpoint = time.time() - start  # time taken for trapezoidal integral

start = time.time()
simpsonIntegral = simpson(f, a, b, Nsimpson)
end = time.time() - start  # time taken for simpson integral

print(
    "Trapezoid Rule Integral: ",
    trapzIntegral,
    "\t Error:",
    (np.pi - trapzIntegral),
    "\t Time: ",
    checkpoint,
    "s",
)
print("Number of Slices: ", Ntrapz)
print("")
print(
    "Simpson Rule Integral: ",
    simpsonIntegral,
    "\t Error:",
    (np.pi - simpsonIntegral),
    "\t Time: ",
    end,
    "s",
)
print("Number of Slices: ", Nsimpson)


print("\n\nPart D")

print("Error Estimation: ", (trapz(f, a, b, 32) - trapz(f, a, b, 16)) / 3)

print("")
