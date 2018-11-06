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

@jit
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
    s_list = [[0 for j in range(steps)] for i in range(steps)]
    i_list = [[0 for j in range(steps)] for i in range(steps)]
    results = [[0 for j in range(steps)] for i in range(steps)]
    print(results)
    s_index = 0
    i_index = 0
    for s in range(1, 10000, int((10000-1)/steps)):
        for i in range(1, 8000, int((8000-1)/steps)):
            result = MonteCarlo(s, i, -2.0, 0.5, -1.25, 1.25)
            results[s_index][i_index] = result
            s_list[s_index][i_index] = s
            i_list[s_index][i_index] = i
            i_index += 1
        s_index += 1
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    res = ax.plot_surface(s_list, i_list, results)

plotConvergence(10)
