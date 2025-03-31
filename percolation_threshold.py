import numpy as np
import matplotlib.pyplot as plt

def percolation(probability,l):
    grid = np.zeros(shape=(l,l))
    for i in range(l):
        for j in range(l):
            u = np.random.uniform()
            if u <= probability:
                grid[i][j] = 1
    t = 2

    for i in range(l):
        if grid[0][i] == 1:
            grid[0][i] = t

    while j < l :
        c = 0
        for j in range(l):
            for i in range(l):
                if grid[i][j] == t:
                    if i-1 >= 0 and grid[i-1][j] == 1:
                        grid[i-1][j] = t+1
                        c += 1
                    if i+1 < l and grid[i+1][j] == 1:
                        grid[i+1][j] = t+1
                        c += 1
                    if j-1 >= 0 and grid[i][j-1] == 1:
                        grid[i][j-1] = t+1
                        c += 1
                    if j+1 < l and grid[i][j+1] == 1:
                        grid[i][j+1] = t+1
                        c += 1
        t += 1
        if c == 0:
            break

    for i in range(l):
        if grid[l-1][i] > 1:
            return 1
    else:   
        return 0
    

def plot_perc(size,T):
    for p in np.arange(0,1,0.05):
        c = 0
        for _ in range(T):
            if percolation(p,size) == 1:
                c += 1
        pflow = c/T
        plt.scatter(p, pflow)
    plt.show()


plot_perc(50,100)