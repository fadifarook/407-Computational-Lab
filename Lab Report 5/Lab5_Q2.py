import numpy as np
import matplotlib.pyplot as plt


"""Code for Q2: Simulated Annealing of Function"""


def f_basic(x, y):
    return x**2 - np.cos(4 * np.pi * x) + (y - 1) ** 2


def getTemp(T0, tau, t):
    return T0 * np.exp(-t / tau)


def draw_normal(sigma):
    """Draw two random numbers from a normal distribution of zero mean and
    standard dev sigma. From Newman section 10.1.6"""
    theta = 2 * np.pi * np.random.random()
    z = np.random.random()
    r = (-2 * sigma**2 * np.log(1 - z)) ** 0.5
    return r * np.cos(theta), r * np.sin(theta)


def decide(newVal, oldVal, temp, bounded=(0, 0, 0, 0)):

    # newval < oldval means always reject
    if np.random.random() > np.exp(-(newVal - oldVal) / temp):
        return 0  # reject (keep same value)
    return 1  # accept  (change value)


# Initial parameters
x0 = 2
y0 = 2

T0 = 100  # initial temperture
Tf = 1e-4  # final temperature
tau = 10000  # cooling schedule

sigma = 1  # standard deviation for motion


def anneal(f, x0, y0, T0, Tf, tau, sigma, bounded=(0, 0, 0, 0)):
    t = 0
    x_array = [x0]
    y_array = [y0]

    T = T0

    reject_counter = 0

    while T > Tf:
        t += 1
        T = getTemp(T0, tau, t)
        dx, dy = draw_normal(sigma)

        x_old, y_old = x_array[-1], y_array[-1]
        xnew, ynew = x_old + dx, y_old + dy

        if bounded != (0, 0, 0, 0):
            if (
                xnew < bounded[0]
                or xnew > bounded[1]
                or ynew < bounded[2]
                or ynew > bounded[3]
            ):
                # reject
                reject_counter += 1
                x_array.append(x_old)
                y_array.append(y_old)
                continue

        if decide(f(xnew, ynew), f(x_old, y_old), T):
            # accept
            x_array.append(xnew)
            y_array.append(ynew)
        else:
            # reject
            x_array.append(x_old)
            y_array.append(y_old)

        # print(x_array[-1], y_array[-1])

    print(reject_counter, tau)

    return x_array, y_array


x_array, y_array = anneal(f_basic, x0, y0, T0, Tf, tau, sigma)
print("Minumum value found at x=", x_array[-1], "y=", y_array[-1])


# Plot x and y
plt.plot(x_array)
plt.plot(y_array)
plt.savefig("Plots/Lab5Q2a.png")
plt.clf()


""" Part B: More complication function"""


def f_complicated(x, y):
    return np.cos(x) + np.cos(np.sqrt(2) * x) + np.cos(np.sqrt(3) * x) + (y - 1) ** 2


# Parameters
x0 = 2
y0 = 2

xmin = 0
xmax = 50
ymin = -20
ymax = 20


x_array, y_array = anneal(
    f_complicated, x0, y0, T0, Tf, tau, sigma, (xmin, xmax, ymin, ymax)
)
print("Minumum value found at x=", x_array[-1], "y=", y_array[-1])

# print(x_array, y_array)

plt.plot(x_array)
plt.plot(y_array)
plt.savefig("Plots/Lab5Q2b.png")
plt.clf()
