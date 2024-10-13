import numpy as np 
import scipy
import matplotlib.pyplot as plt
from gaussxw import gaussxwab, gaussianQuadrature

"""Realtivistic Spring Particle"""




"""Part A"""

def g(x, k, m, x0):
    """Function for v = dx/dt"""

    c = 3e8

    temp_val = (2*m*c**2) + (k * (x0**2 - x**2) / 2)
    temp_val *= k * (x0**2 - x**2)

    temp_val /= 2 * ((m*c**2) + (k * (x0**2 - x**2) / 2))**2

    return c * temp_val**0.5


def T(x, k, m, x0):
    """Integrand"""
    return 4 / g(x, k, m, x0)

m = 1  #[kg]
k = 12  #[N/m]

x0 = 0.01  # [m]


N1 = 8
N2 = 16

# xp1, wp1 = gaussxwab(N1, 0, x0)
# xp2, wp2 = gaussxwab(N2, 0, x0)

# s1 = 0.0
# for  i in range(N1):
#     s1 += wp1[i] * T(xp1[i], k, m, x0)

# s2 = 0.0
# for  i in range(N2):
#     s2 += wp2[i] * T(xp2[i], k, m, x0)

xp, wp = gaussxwab(N1, 0, x0)
xp1, wp1, s1 = gaussianQuadrature(T, N1, 0, x0, k, m, x0)
xp2, wp2, s2 = gaussianQuadrature(T, N2, 0, x0, k, m, x0)

print(f"N = 8 : {s1} \t N = 16 {s2}")


print(wp1, wp)


"""Part B: plot T"""

x_array = np.linspace(0, x0, 100)

plt.plot(x_array, T(x_array, k, m, x0), label = 'T')
plt.scatter(xp1, wp1 * T(xp1, k, m, x0), label='8')
plt.scatter(xp2, wp2 * T(xp2, k, m, x0), label='16')

plt.legend()
plt.show()





"""Part C: T as a function of x0"""

c = 3e8
k = 12
m = 1
xc = c * np.sqrt(m/k)
x0_array = np.linspace(1, 10*xc, 50)

N = 16

s_array = np.zeros(len(x0_array))

for i in range(len(x0_array)):
    s_array[i] = gaussianQuadrature(T, N, 0, x0_array[i], k, m, x0_array[i])[2]

plt.plot(x0_array, s_array)
plt.axhline(2*np.pi * np.sqrt(m/k), color='green')
plt.plot(x0_array, 4*x0_array/c)
plt.show()
