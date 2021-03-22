import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from cluster_kmeans import *

debug = False
k = 5
case = 2

if case == 1:
    # Case 1:  Get some 2D data from sklearn
    from sklearn.datasets import make_blobs
    features, true_labels = make_blobs(n_samples=300,centers=k,cluster_std=2.75,random_state=42)
    if debug: print (true_labels)
    df = pd.DataFrame(features, columns = ['X','Y'])
else:
    if case == 2:
        # Case 2:  Produce some 3D spheres filled with datapoints
        features = np.zeros((500,3))
        import random
        xc = np.array([[0,0,0],[1,1,1],[-1,-1,-1],[1,-1,-1],[-1,1,1]])

        radius = 0.2
        for j in range(100):
            for i in range(5):
                x = random.uniform(-radius,radius)
                y = random.uniform(-radius,radius)
                z = random.uniform(-radius,radius)
                v = np.array([xc[i][0]+x,xc[i][1]+y,xc[i][2]+z])
                index = 5*j+i
                features[5*j+i] = v
        df = pd.DataFrame(features, columns=['X', 'Y','Z'])

xcenter_index, sse = cluster_kmeans(df,k)
df['Index'] = xcenter_index

if case == 1:
    plt.scatter(df['X'],df['Y'],c=df['Index'])
else:
    if case == 2:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df['X'],df['Y'],df['Z'],c=df['Index'])

plt.show()