import numpy as np
import matplotlib.pyplot as plt
import os
import time


print("Part B")

# Gets the absolute path of cdata.txt
currentDir = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(currentDir, 'cdata.txt')


def std_1(array):
    """Function corresponding to Standard Deviation Equation 1 (Two pass)"""

    mean = np.mean(array)
    temporary_value = np.sum((array-mean)**2) # intermediate step

    return np.sqrt(temporary_value / (len(array) - 1))


def std_2(array):
    """Function corresponding to Standard Deviation Equation 2 (One pass)"""

    mean = np.mean(array)

    temporary_value = np.sum(array**2)
    temporary_value -= len(array) * mean**2

    if temporary_value < 0:
        print("Negative Square Root")  # Negative check
        temporary_value *= -1

    return np.sqrt(temporary_value / (len(array) - 1))


# Load the data from the file
speed = np.loadtxt(filename)

true_value = np.std(speed, ddof=1)
std1_value = std_1(speed)
std2_value = std_2(speed)

print("Numpy Standard Deviation: ", true_value)
print("Equation 1 (two pass) Standard Deviation: ", std1_value, '\t Relative Error: ', (true_value - std1_value) / true_value)
print("Equation 2 (one pass) Standard Deviation: ", std2_value, '\t Relative Error: ', (true_value - std2_value) / true_value)





print("\n\nPart C")

# Create an array with normal distribuion with same sigma and number of values (n)
# But different mean1 and mean2
mean1 = 0.
mean2 = 1.e7
sigma = 1.
n = 2000

# Create arrays
normalSequence1 = np.random.normal(mean1, sigma, 2000)
normalSequence2 = np.random.normal(mean2, sigma, 2000)


true_value1 = np.std(normalSequence1, ddof=1)
std1_value_mean1 = std_1(normalSequence1)
std2_value_mean1 = std_2(normalSequence1)

print("Mean = 0")
print("True Standard Deviation", true_value1)
print("Equation 1 (Two Pass) Standard Deviation: ", std1_value_mean1, "\t Relative Error", (std1_value_mean1 - true_value1)/ true_value1)
print("Equation 2 (One Pass) Standard Deviation: ", std2_value_mean1, "\t Relative Error", (std2_value_mean1 - true_value1)/ true_value1)

print("\nMean = 1e7")
true_value2 = np.std(normalSequence2, ddof=1)
std1_value_mean2 = std_1(normalSequence2)
std2_value_mean2 = std_2(normalSequence2)
print("True Standard Deviation", true_value2)
print("Equation 1 (Two Pass) Standard Deviation: ", std1_value_mean2, "\t Relative Error", (std1_value_mean2 - true_value2)/ true_value2)
print("Equation 2 (One Pass) Standard Deviation: ", std2_value_mean2, "\t Relative Error", (std2_value_mean2 - true_value2)/ true_value2)








print("\n\nPart D")

def std_3(array):
    """New standard deviation calculation where we subtract a random value"""

    # Subtract a random value beforehand
    randomValue = np.random.choice(array)
    array -= randomValue

    mean = np.mean(array)

    temporary_value = np.sum(array**2)
    temporary_value -= len(array) * mean**2

    if temporary_value < 0:
        print("Negative Square Root")  # Negative check
        temporary_value *= -1

    return np.sqrt(temporary_value / (len(array) - 1))


print("With modified standard deviation calculation:")
std3_value_mean1 = std_3(normalSequence1)
std3_value_mean2 = std_3(normalSequence2)
print("Mean = 0 ; New Standard Deviation: ", std3_value_mean1, "\t Relative Error: ", (std3_value_mean1 - true_value1)/ true_value1)
print("Mean = 1e7 ; New Standard Deviation: ", std3_value_mean2, "\t Relative Error: ", (std3_value_mean2 - true_value2)/ true_value2)






print("""\n\nQuestion 2""")


def p(u):
    return (1-u)**8

def q(u):
    return 1 - 8*u + 28*u**2 - 56*u**3 + 70*u**4 - 56*u**5 + 28*u**6 - 8*u**7 + u**8

def xterms(u):
    return np.array([(1-u)**8, -1, 8*u, -28*u**2, 56*u**3 , -70*u**4 , 56*u**5, -28*u**6, 8*u**7, -u**8])


N = 500
u_array = np.linspace(0.98, 1.02, N)

plt.plot(u_array ,p(u_array), color='red')
plt.plot(u_array ,q(u_array), color='blue')  # noisier
# plt.show()


difference_array = p(u_array) - q(u_array)
plt.plot(u_array, difference_array, color='green')
plt.show()

"""Plot histogram"""
plt.hist(difference_array, 100)
plt.show()

C = 1e-16
numTerms = 10
terms = xterms(u=1)
squaredTerms = np.mean(terms**2)
print("Standard Deviation = ", np.std(difference_array))
print("Calculated Standard Deviation = ", C * np.sqrt(numTerms) * np.sqrt(squaredTerms))





"""2C"""
u_array = np.linspace(0.980, 0.984, 50)
plt.plot(u_array, abs(p(u_array) - q(u_array))/abs(p(u_array)))
plt.show()

nearTerms = xterms(u=0.983)
squarednearTerms = np.mean(nearTerms**2)

fractional_error = C * np.sqrt(squarednearTerms) / (np.sqrt(numTerms) * np.mean(nearTerms))

print(fractional_error)





""" 2D """

def func_multiply(u):
    return (u**8)/((u**4)*(u**4))

N = 500
u_array = u_array = np.linspace(0.98, 1.02, N)

plt.plot(u_array, func_multiply(u_array)-1.)
plt.show()


print("Standard Deviation: ", np.std(func_multiply(u_array)-1.))
print("Caluclated STD: ", C)


print("""\n\nQuestion 3""")

def f(x):
    return 4 / (1+x**2)  # answer is Pi

def trapz(func:callable, start, end, N):
    """Trapezoidal integration function"""
    h = (end-start)/N
    s = (func(end) + func(start)) / 2

    for i in range(1, N):  # dont want i == 0
        s += func(start + i*h)

    return s*h


def simpson(func, start, end, N):
    h = (end-start)/N
    s = func(start) + func(end)

    for i in range(1, N):

        if i%2 == 0:
            s += 2 * func(start + i*h)
        
        else:
            s += 4 * func(start + i*h)



    return s*h/3


Ntrapz = 4096  # number of slices
Nsimpson = 16  # number of slices
a = 0
b = 1

start = time.time()
trapzIntegral = trapz(f, a, b, Ntrapz)
checkpoint = time.time() - start
simpsonIntegral = simpson(f, a, b, Nsimpson)
end = time.time() - checkpoint

print("Trapezoid Rule Integral: ", trapzIntegral, "\t Error:", (np.pi - trapzIntegral), "\t Time: ", checkpoint)
print("Simpson Rule Integral: ", simpsonIntegral, "\t Error:", (np.pi - simpsonIntegral), "\t Time: ", end)





"""Practical Error Estimation"""
print("\n")

print("Error Estimation: ", (trapz(f, a, b, 32) - trapz(f, a, b, 16))/3)
print("Error Estimation: ", (simpson(f, a, b, 32) - simpson(f, a, b, 16))/15)
print("Error Estimation: ", (simpson(f, a, b, 64) - simpson(f, a, b, 32))/15)