import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from numba import jit
import random
from mpl_toolkits.mplot3d import Axes3D


@jit
def mandelbrot(z, maxiter, horizon, log_horizon):
    c = z
    for n in range(maxiter):
        az = abs(z)
        if az > horizon:
            return n - np.log(np.log(az)) / np.log(2) + log_horizon
        z = z * z + c
    return 0

@jit
def checkInMS(z, maxiter, horizon, log_horizon):
    #     print(z)
    if mandelbrot(z, maxiter, horizon, log_horizon) == 0:
        return 1.0
    return 0.0


def MonteCarlo(s, i, xmin, xmax, ymin, ymax):
    horizon = 2.0**40
    log_horizon = np.log(np.log(horizon)) / np.log(2)
    area = (xmax - xmin) * (ymax - ymin)
    ctr = 0
    for j in range(s):
        x = xmin + (xmax - xmin) * random.random()
        y = ymin + (ymax - ymin) * random.random()
        z = x + 1j * y
        ctr += checkInMS(z, i, horizon, log_horizon)
    return (ctr / s) * area


def plotConvergence(steps):
    x = np.linspace(1, 10000, steps + 1).astype('int')
    y = np.linspace(1, 8000, steps + 1).astype('int')
    results = [[0 for j in range(steps+1)] for i in range(steps+1)]
    X, Y = np.meshgrid(x, y)
    s_in = 0
    i_in = 0
    for s in x:
        for i in y:
            result = MonteCarlo(s, i, -2.0, 0.5, -1.25, 1.25)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    res = ax.plot_surface(x, y, results)

plotConvergence(10)
