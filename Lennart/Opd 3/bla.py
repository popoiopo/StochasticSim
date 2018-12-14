import numpy as np
import scipy.spatial as sp
import matplotlib.pyplot as plt
import tkinter
import pandas as pd
import math
from progress.bar import Bar
from progress.bar import IncrementalBar
from numba import jit
from scipy.spatial import distance_matrix
import time
from IPython.display import IFrame, display, HTML, Markdown
import random

a = [100]
a2 = [100]
a3 = [100]
b = 100
c = 100
d = 100
for i in range(1000):
    b *= 0.95
    c *= 0.99
    d *= 0.995
    a.append(b)
    a2.append(c)
    a3.append(d)

plt.plot(a)
plt.plot(a2)
plt.plot(a3)
plt.show()