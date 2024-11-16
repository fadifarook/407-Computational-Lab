import numpy as np
import matplotlib.pyplot as plt


# Initial Parameters
epsilon = 1
deltax = 0.02  # [m]
deltat = 0.005  # [s]
Lx = 2 * np.pi  # [m]

Tf = 2  # [s]


# Initial arrays (t = 0)
x = np.arange(0, Lx, deltax)
N = len(x) - 1
u = np.sin(x)

# Plot of u vs x at time = 0
plt.plot(x, u)
plt.xlabel("x [m]")
plt.ylabel("u [m/s]")
plt.title("u vs x at time 0")
plt.savefig("Lab4Q3a.png")
plt.clf()


"""DONT FORGET u BOUNDARY CONDISH"""


"""Implement FTCS method"""

beta = epsilon * deltat / deltax

# Time loop
epsilon = deltat / 100
t1 = 0.5  # [s]
t2 = 1  # [s]
t3 = 1.5  # [s]
tend = Tf + epsilon

t = 0

# define two arrays for u
u1 = u
u2 = u - deltat * u * np.cos(x)

while t < tend:
    t = t + deltat

    # Boundary condition satisfy
    u1[0], u2[0], u1[-1], u2[-1] = 0, 0, 0, 0

    u_new = np.zeros(len(x))
    u2_squared = u2**2

    u_new[1:N] = u1[1:N] - beta / 2 * (u2_squared[2 : N + 1] - u2_squared[0 : N - 1])

    u1, u2 = u2.copy(), u_new.copy()

    # Plot of u vs x at specified times
    if abs(t - t1) < epsilon:
        print(x[np.argmax(u2)])
        plt.plot(x, u2)
        plt.xlabel("x [m]")
        plt.ylabel("u [m/s]")
        plt.title("u vs x at time " + str(t1))
        plt.savefig("Lab4Q3b.png")
        plt.clf()

    if abs(t - t2) < epsilon:
        print(x[np.argmax(u2)])
        plt.plot(x, u2)
        plt.xlabel("x [m]")
        plt.ylabel("u [m/s]")
        plt.title("u vs x at time " + str(t2))
        plt.savefig("Lab4Q3c.png")
        plt.clf()

    if abs(t - t3) < epsilon:
        print(x[np.argmax(u2)])
        plt.plot(x, u2)
        plt.xlabel("x [m]")
        plt.ylabel("u [m/s]")
        plt.title("u vs x at time " + str(t3))
        plt.savefig("Lab4Q3d.png")
        plt.clf()
