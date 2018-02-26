import numpy as np

p = 2
k = 3
Sij = np.zeros([3,4,2])

for i in range(3):
    for j in range(4):
        Sij[i,j,0] = i
        Sij[i,j,1]=j
print Sij


r = np.random.random([2,3])
print r