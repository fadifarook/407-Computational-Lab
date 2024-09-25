import numpy as np
import matplotlib.pyplot as plt
import os


# Done due to issues in opening file in the same path
currentDir = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(currentDir, 'cdata.txt')




def std_1(array):

    mean = np.mean(array)

    temporary_value = np.sum((array-mean)**2)

    return np.sqrt(temporary_value / (len(array) - 1))

def std_2(array):
    
    mean = np.mean(array)

    temporary_value = np.sum(array**2)

    # for i in range(len(array)):
    #     temporary_value += array[i]**2

    temporary_value -= len(array) * mean**2

    if temporary_value < 0:
        print("Negative Square Root")
        temporary_value *= -1

    
    return np.sqrt(temporary_value / (len(array) - 1))



speed = np.loadtxt(filename)

true_value = np.std(speed, ddof=1)
std1_value = std_1(speed)
std2_value = std_2(speed)

print(true_value, std1_value)
print(true_value, std2_value)

print((true_value - std1_value) / true_value)
print((true_value - std2_value) / true_value)





"""Question 1C"""
mean1 = 0.
mean2 = 1.e7
sigma = 1.
n = 2000

normalSequence1 = np.random.normal(mean1, sigma, 2000)
normalSequence2 = np.random.normal(mean2, sigma, 2000)

true_value1 = np.std(normalSequence1, ddof=1)
print("One Pass Rel Error", (std_1(normalSequence1) - true_value1)/ true_value1)
print("Two Pass Rel Error", (std_2(normalSequence1) - true_value1)/ true_value1)

true_value2 = np.std(normalSequence2, ddof=1)
print("One Pass Rel Error", (std_1(normalSequence2) - true_value2)/ true_value2)
print("Two Pass Rel Error", (std_2(normalSequence2) - true_value2)/ true_value2)









print("""\n\nQuestion 2""")


def p(u):
    return (1-u)**8

def q(u):
    return 1 - 8*u + 28*u**2 - 56*u**3 + 70*u**4 - 56*u**5 + 28*u**6 - 8*u**7 + u**8



N = 500
u_array = np.linspace(0.98, 1.02, N)

plt.plot(u_array ,p(u_array), color='red')
plt.plot(u_array ,q(u_array), color='blue')  # noisier
# plt.show()


difference_array = p(u_array) - q(u_array)
plt.plot(u_array, difference_array, color='green')
# plt.hist(difference_array)
# plt.show()

C = 1e-16
numTerms = 10
terms = np.array([0, -1, 8, -28, 56, -70, 56, -28, 8, -1])
lastTerm = np.mean(terms**2)
print("Standard Deviation = ", np.std(difference_array))
print("Calculated Standard Deviation = ", C * np.sqrt(numTerms) * np.sqrt(lastTerm))





"""2C"""
u_array = np.linspace(0.980, 0.984, 50)
plt.plot(u_array, abs(p(u_array) - q(u_array))/abs(p(u_array)))
# plt.show()

print(np.mean(terms))

fractional_error = C * np.sqrt(lastTerm) / (np.sqrt(numTerms) * np.mean(terms))

print(fractional_error)





""" 2D """





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

trapzIntegral = trapz(f, a, b, Ntrapz)
simpsonIntegral = simpson(f, a, b, Nsimpson)
print("Trapezoid Rule Integral: ", trapzIntegral, "\t Error:", (np.pi - trapzIntegral))
print("Simpson Rule Integral: ", simpsonIntegral, "\t Error:", (np.pi - simpsonIntegral))
