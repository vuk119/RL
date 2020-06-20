"""

Some useful plot functions

"""

import matplotlib.pyplot as plt
import numpy as np


def matrix_surf(m, xlimits=None, ylimits=None, **kwargs):
    if xlimits is None:
        xlimits = [0, m.shape[0]]
    if ylimits is None:
        ylimits = [0, m.shape[1]]

    Y, X = np.meshgrid(np.arange(ylimits[0], ylimits[1]), np.arange(xlimits[0], xlimits[1]))

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d',**kwargs)
    ax.plot_surface(X,Y,m)
    plt.show()

def matrix_scatter(m):
    X=[]
    Y=[]
    Z=[]
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            X.append(i)
            Y.append(j)
            Z.append(m[i,j])


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z)
    plt.show()



# mat = np.zeros((6,5))
# mat[0,0] = 5
# mat[0,1] = 4
# mat[1,0] = 4
# mat[1,1] = 3
# mat[1,2] = 3
# mat[2,1] = 3
# mat[0,2] = 3
# mat[2,0] = 3
# mat[0,3] = 3
# mat[3,0] = 3
# matrix_surf(mat, xlabel = 'X AXIS', ylabel = 'Y AXIS', zlabel='Z', xticks =range(10))
#
#
#
# Y, X = np.meshgrid(np.arange(mat.shape[1]), np.arange(mat.shape[0]))
#
# print(X)
# print(Y)
