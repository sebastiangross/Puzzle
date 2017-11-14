__author__ = 'Herakles II'

import numpy as np
import matplotlib.pyplot as plt

from source import create_random_puzzle

N = 10000
size = 100


types = np.zeros(N)
sides = np.zeros(shape=(N, size, size))

for n in range(N):
    sides[n, :, :], types[n] = create_random_puzzle.create_random_side(size)

filled = (sides>0).mean(axis=1).mean(axis=1)
mid_filled = (sides[:, size/3:size*2/3, size/3:size*2/3]>0).mean(axis=1).mean(axis=1)

min = np.zeros(N)
max = np.zeros(N)
for n in range(N):
    b = np.abs(sides[n, 1:, 1:]-sides[n, :-1, :-1])
    ind = np.where(b>0)
    s = np.true_divide(ind[0]+size-ind[1], 2*size)
    if len(s) > 0:
        max[n] = np.min(s)
        min[n] = np.max(s)
    else:
        max[n] = 0.5
        min[n] = 0.5

plt.scatter(min, max, c=types)
plt.show()

# Result:
# if max >0.55
# then type = 1
# if min < 0.45
# then type = -1
# else type = 0