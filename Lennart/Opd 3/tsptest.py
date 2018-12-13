import numpy as np
import scipy.spatial as sp
import matplotlib.pyplot as plt
import tkinter
import pandas as pd
import math
from progress.bar import Bar
from progress.bar import IncrementalBar
from numba import jit

def prettyfie(ax, x, y, title, legendYN='Yes'):

    # Remove borders
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(alpha=0.25)

    # Remove unnecessary ticks
    ax.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        labelbottom=True,
        left=False,
        right=False,
        labelleft=True)

    if legendYN == 'Yes':
        # Create legend and grid
        ax.legend(framealpha=1, shadow=True)
        ax.legend()

    # Set labels and title
    ax.set_xlabel(x, fontsize=14)
    ax.set_ylabel(y, fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)


@jit
def getScheme_data(iters, plot="No", scheme="Sigmoid", markovChain=1):

    answer = []

    if scheme == "Sigmoid":
        linspace = np.linspace(0, 5, iters)
        for x in linspace:
            answer.append(0.5 +
                          math.erf((math.sqrt(math.pi) / 2) * (x - 2)) * -0.5)
    elif scheme == "Sigmoidvar":
        linspace = np.linspace(0, 4, iters)
        for x in linspace:
            answer.append(0.5 +
                          math.erf((math.sqrt(math.pi) / 2) * (x - 2)) * -0.5)
    elif scheme == "Exp":
        for x in range(iters):
            answer.append(np.exp(-x / (iters / 10)))
    elif scheme == "Expvar":
        for x in range(iters):
            answer.append(
                max(
                    np.exp(-x / iters) -  1.05 * (1 / math.e), #((1.5 * x) / iters)
                    0))
    elif scheme == "Hillclimb":
        for x in range(iters):
            answer.append(0)
    elif scheme == "Binary":
        for x in range(iters):
            if x < iters / 2:
                answer.append(1)
            else:
                answer.append(0)
    elif scheme == "Linear":
        for x in range(iters):
            answer.append(1.0 - x / iters)

    if plot == "Yes":
        plotanswer = np.repeat(answer, markovChain)
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        ax.plot(plotanswer, label="Acceptance chance")
        ax.legend()
        prettyfie(ax, "iteration", "Acceptance", scheme, legendYN="Yes")
        plt.show()
    return np.array(answer)

@jit
def spawn(n):
    pointsdict = {}
    points = []
    for i in range(n):
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        p = [x, y]
        pointsdict[i] = p
        points.append(i)
    return points, pointsdict

@jit
def scorecalc(n, points, pointsdict):
    difs = 0
    for i in range(n - 1):
        a = pointsdict[points[i]]
        b = pointsdict[points[i + 1]]
        dif =  math.hypot(b[0]- a[0], b[1] - a[1])
        difs += dif
    a = pointsdict[points[n-1]]
    b = pointsdict[points[0]]
    dif =  math.hypot(b[0]- a[0], b[1] - a[1])
    difs += dif
    return difs


@jit
def anneal(n, points, pointsdict, score, temp):
    new = list(np.copy(points))
    b1 = np.random.randint(0, len(points))
    pp = points[b1]
    nnn = max(3, int(temp * n))
    b2 = np.random.randint(0, len(points))
    switch = np.random.random()
    ppp = new.pop(b1)
    new.insert(b2, pp)
    # if switch < 0.5:
    #     new = np.concatenate((np.array(points[:b1], int), np.array(points[b1:b2][::-1], int), np.array(points[b2:], int)))
    # else:
    #     new = np.concatenate((np.array(points[b2:][::-1], int), np.array(points[b1:b2], int), np.array(points[:b1][::-1], int)))
    # print(new)
    newscore = scorecalc(n, new, pointsdict)
    if temp > 0:
        chance = np.exp(-(newscore - score) / (20 * temp))
    else:
        chance = 0
    if newscore < score:
        return new, newscore
    elif np.random.uniform(0, 1) < chance:
        return new, newscore
    return points, score

@jit
def plot(n, points, pointsdict):
    x = []
    y = []
    for i in pointsdict:
        x.append(pointsdict[i][0])
        y.append(pointsdict[i][1])
    plt.scatter(x[1:], y[1:], color="red")
    plt.scatter(x[0], y[0], color="blue")
    for i in range(n - 1):
        a = pointsdict[points[i]]
        b = pointsdict[points[i + 1]]
        x = [a[0], b[0]]
        y = [a[1], b[1]]
        plt.plot(x, y, color="black")
    a = pointsdict[points[n - 1]]
    b = pointsdict[points[0]]
    x = [a[0], b[0]]
    y = [a[1], b[1]]
    plt.plot(x, y, color="black")
    plt.show()

@jit
def read_tsp_file(name):
    nodes = pd.read_csv('%s.tsp.txt' % name, skiprows=6, skipfooter=1, delim_whitespace=True, header=None, names=('id', 'x', 'y') ,engine='python')
    ids = nodes['id']
    xs = nodes['x']
    ys = nodes['y']
    points = list(range(len(ids)))
    pointsdict = {}
    for i in ids:
        pointsdict[i-1] = [xs[i-1], ys[i-1]]
    return points, pointsdict

@jit
def read_opt_file(name):
    route = pd.read_csv('%s.opt.tour.txt' % name, skiprows=4, skipfooter=1, header=None, delim_whitespace=True, names=('id',), engine='python')
    ids = route['id']
    points = list(range(len(ids) - 1))
    for i in range(len(ids) - 1):
        points[i] = int(ids[i]) - 1
    return points
        


@jit
def run(n, iterations, scheme, markovChain, plotScheme="No", pointsdict=None, points=None):
    if pointsdict == None:
        points, pointsdict = spawn(n)
    score = scorecalc(n, points, pointsdict)
    temp = 1.0
    steps = []
    checklist = []
    schemeData = np.repeat(
        getScheme_data(math.ceil(iterations / markovChain), plot=plotScheme, scheme=scheme, markovChain=markovChain),
        markovChain)
    maxiterations = iterations
    bar = IncrementalBar('Processing', max=maxiterations)
    for i in range(iterations):
        temp = schemeData[i]
        points, scorenew = anneal(n, points, pointsdict, score, temp)
        steps.append(points)
        checklist.append(scorecalc(n, points, pointsdict))
        score = scorenew
        bar.next()
    bar.finish()
    print("finished")
    print("Final score = {}".format(score))
    plot(n, points, pointsdict)
    return points, score, steps, pointsdict, checklist

# schemes = ["Sigmoid", "Exp", "Expvar", "Hillclimb"]
# schemes = ["Sigmoidvar", "Expvar", "Hillclimb"]
# schemes = ["Expvar"]
schemes = ["Linear"]

# points, pointsdict = spawn(n)
points, pointsdict = read_tsp_file('a280')
n = len(points)
print(n)


checklists = []
stepss = []
for scheme in schemes:
    point, score, steps, pointsdict, checklist = run(n, 500000, scheme, 10000, plotScheme="Yes", pointsdict=pointsdict, points=points)
    checklists.append(checklist)
    stepss.append(steps)
   
[plt.plot(check, label=schemes[i]) for i, check in enumerate(checklists)]
plt.legend()
plt.show()


optimal = read_opt_file('a280')
optscore = scorecalc(len(optimal), optimal, pointsdict)
print(optscore)
plot(n, optimal, pointsdict)
# point, score, steps, pointsdict = run(n, 10000)