import numpy as np
from gaussxw import gaussianQuadrature, gaussxwab
import matplotlib.pyplot as plt

def f(x):
    return x**2

N = 8
x_array = np.linspace(0, 0.01, 100)

xp, wp, s = gaussianQuadrature(f, N, 0, 0.01)

xp, wp2 = gaussxwab(N, 0, 0.01)

print(s, np.trapz(f(x_array), x_array))


print(wp)

# plt.plot(x_array, f(x_array))
plt.scatter(xp, wp * f(xp))
plt.show()

