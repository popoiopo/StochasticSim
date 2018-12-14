#!/usr/bin/env python

""" Traveling salesman problem solved using Simulated Annealing.
"""
from scipy import *
from pylab import *
import pandas as pd

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

def getScheme_data(iters, plot="No", scheme="Sigmoid", markovChain=1):

    answer = []

    if scheme == "Sigmoid":
        linspace = np.linspace(0, 5, iters)
        for x in linspace:
            answer.append(16 * (0.5 +
                          math.erf((math.sqrt(math.pi) / 2) * (x - 2)) * -0.5))
    elif scheme == "Sigmoidvar":
        linspace = np.linspace(0, 4, iters)
        for x in linspace:
            answer.append(16 * (0.5 +
                          math.erf((math.sqrt(math.pi) / 2) * (x - 2)) * -0.5))
    elif scheme == "Exp":
        for x in range(iters):
            answer.append(16 * np.exp(-x / (iters / 10)))
    elif scheme == "Expvar":
        for x in range(iters):
            answer.append(16 * 
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
            answer.append(16.0 - 16 * x / iters)
    elif scheme == "Linearvar":
        for x in range(iters):
            answer.append(0.5 - 0.5 * x / iters)

    if plot == "Yes":
        plotanswer = np.repeat(answer, markovChain)
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        ax.plot(plotanswer, label="Acceptance chance")
        ax.legend()
        prettyfie(ax, "iteration", "Acceptance", scheme, legendYN="Yes")
        plt.show()
    return np.array(answer)

def Distance(R1, R2):
    return sqrt((R1[0]-R2[0])**2+(R1[1]-R2[1])**2)

def TotalDistance(city, R):
    dist=0
    for i in range(len(city)-1):
        dist += Distance(R[city[i]],R[city[i+1]])
    dist += Distance(R[city[-1]],R[city[0]])
    return dist
    
def reverse(city, n):
    nct = len(city)
    nn = int((1+ ((n[1]-n[0]) % nct))/2) # half the lenght of the segment to be reversed
    # the segment is reversed in the following way n[0]<->n[1], n[0]+1<->n[1]-1, n[0]+2<->n[1]-2,...
    # Start at the ends of the segment and swap pairs of cities, moving towards the center.
    city = list(city)
    for j in range(nn):
        k = (n[0]+j) % nct
        l = (n[1]-j) % nct
        (city[k],city[l]) = (city[l],city[k])  # swap
    return city
    
def transpt(city, n):
    nct = len(city)
    
    newcity=[]
    # Segment in the range n[0]...n[1]
    for j in range( (n[1]-n[0])%nct + 1):
        newcity.append(city[ (j+n[0])%nct ])
    # is followed by segment n[5]...n[2]
    for j in range( (n[2]-n[5])%nct + 1):
        newcity.append(city[ (j+n[5])%nct ])
    # is followed by segment n[3]...n[4]
    for j in range( (n[4]-n[3])%nct + 1):
        newcity.append(city[ (j+n[3])%nct ])
    return newcity

def read_tsp_file(name):
    nodes = pd.read_csv('%s.tsp.txt' % name, skiprows=6, skipfooter=1, delim_whitespace=True, header=None, names=('id', 'x', 'y') ,engine='python')
    ids = nodes['id']
    xs = nodes['x']
    ys = nodes['y']
    points = list(range(len(ids)))
    pointsdict = {}
    for i in ids:
        points[i-1] = [xs[i-1], ys[i-1]]
    return points

def Plot(city, R, dist):
    # Plot
    Pt = [R[city[i]] for i in range(len(city))]
    Pt += [R[city[0]]]
    Pt = array(Pt)
    title('Total distance='+str(dist))
    plot(Pt[:,0], Pt[:,1], color="black")
    scatter(Pt[:,0], Pt[:,1], color="red")
    show()

def run(iterations, stopdict, scheme="Exp", mcl=1):

    ncity = 280        # Number of cities to visit
    maxTsteps = int(iterations / mcl)    # Temperature is lowered not more than maxTsteps
    Tstart = 0.2       # Starting temperature - has to be high enough
    fCool = 0.9        # Factor to multiply temperature at each cooling step
    maxSteps = mcl     # Number of steps at constant temperature
    maxAccepted = mcl / 10   # Number of accepted steps at constant temperature

    Preverse = 0.5      # How often to choose reverse/transpose trial move


    R = read_tsp_file('a280')
    R = array(R)

    # The index table -- the order the cities are visited.
    city = list(range(ncity))
    # Distance of the travel at the beginning
    dist = TotalDistance(city, R)

    # Stores points of a move
    n = zeros(6, dtype=int)
    nct = len(R) # number of cities
    
    

    # Plot(city, R, dist)

    schemeData = np.repeat(
        getScheme_data(math.ceil(maxTsteps), plot="Yes", scheme=scheme, markovChain=1),
        1)

    T = schemeData[0] # temperature
    distold = 100000
    counter = stopdict[mcl]
    
    dists = []
    for t in range(maxTsteps):  # Over temperature

        accepted = 0
        for i in range(maxSteps): # At each temperature, many Monte Carlo steps
            
            while True: # Will find two random cities sufficiently close by
                # Two cities n[0] and n[1] are choosen at random
                n[0] = int((nct)*rand())     # select one city
                n[1] = int((nct-1)*rand())   # select another city, but not the same
                if (n[1] >= n[0]): n[1] += 1   #
                if (n[1] < n[0]): (n[0],n[1]) = (n[1],n[0]) # swap, because it must be: n[0]<n[1]
                nn = (n[0]+nct -n[1]-1) % nct  # number of cities not on the segment n[0]..n[1]
                if nn>=3: break
        
            # We want to have one index before and one after the two cities
            # The order hence is [n2,n0,n1,n3]
            n[2] = (n[0]-1) % nct  # index before n0  -- see figure in the lecture notes
            n[3] = (n[1]+1) % nct  # index after n2   -- see figure in the lecture notes
            
            if Preverse > rand(): 
                # Here we reverse a segment
                # What would be the cost to reverse the path between city[n[0]]-city[n[1]]?
                de = Distance(R[city[n[2]]],R[city[n[1]]]) + Distance(R[city[n[3]]],R[city[n[0]]]) - Distance(R[city[n[2]]],R[city[n[0]]]) - Distance(R[city[n[3]]],R[city[n[1]]])
                
                chance = 0
                if T != 0:
                    chance = exp(-de/T)
                if de<0 or chance>rand(): # Metropolis
                    accepted += 1
                    dist += de
                    city = reverse(city, n)
            else:
                # Here we transpose a segment
                nc = (n[1]+1+ int(rand()*(nn-1)))%nct  # Another point outside n[0],n[1] segment. See picture in lecture nodes!
                n[4] = nc
                n[5] = (nc+1) % nct
        
                # Cost to transpose a segment
                de = -Distance(R[city[n[1]]],R[city[n[3]]]) - Distance(R[city[n[0]]],R[city[n[2]]]) - Distance(R[city[n[4]]],R[city[n[5]]])
                de += Distance(R[city[n[0]]],R[city[n[4]]]) + Distance(R[city[n[1]]],R[city[n[5]]]) + Distance(R[city[n[2]]],R[city[n[3]]])
                
                chance = 0
                if T != 0:
                    chance = exp(-de/T)
                if de<0 or chance>rand(): # Metropolis
                    accepted += 1
                    dist += de
                    city = transpt(city, n)
            
            dists.append(dist)
                    
            if accepted > maxAccepted: break

        # Plot
        # Plot(city, R, dist)
            
        # print("T=%10.5f , distance= %10.5f , counter= %d" %(T, dist, counter))
        # T *= fCool             # The system is cooled down
        T = schemeData[t]

        if dist == distold:
            counter -= 1
        else:
            counter = stopdict[mcl]
        if counter == 0:
            break
        distold = dist
        
        # if accepted == 0: break  # If the path does not want to change any more, we can stop

    print("distance= %10.5f" %(dist))
    Plot(city, R, dist)
    
    plt.plot(dists)
    plt.show()
    return dists


iterations = 10000000
schemes = ["Hillclimb"]
mcls = [500, 10000, 50000]
stopdict = {500: 200, 10000: 100, 50000: 50}


checklists = []
data = []
rowcount = 0
for scheme in schemes:
    data.append([])
    for mcl in mcls:
        print("## Scheme= {},     markov length = {}\n".format(scheme, mcl))
        checklist = run(iterations, stopdict, scheme=scheme, mcl=mcl)
        checklists.append(checklist)
        data[rowcount].append(checklist[-1])
    rowcount += 1








schemas = []
mcls = []
for i, scheme in enumerate(schemes):
    schemas.append(scheme)
arrays = [schemas]
tuples = list(zip(*arrays))
# index = pd.MultiIndex.from_tuples(tuples, names=['Scheme', 'Change method'])
s = pd.DataFrame(data, index='Scheme', columns=mcl)
s.T
print(s.T)