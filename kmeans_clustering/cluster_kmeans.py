import numpy as np
import pandas as pd

def SSE(df,xcenter,xc_index):

    m = len(df)
    n = len(df.columns)
    sse = 0

    for i in range(m):
        for j in range(n):
            sse = sse + (df.iloc[:,j][i]-xcenter[xc_index[i]][j])**2

    return sse

def xcenter_index(df,xcenter,k):

    ########## Assign each data point to a center, based on Euclidean distance
    m = len(df)
    n = len(df.columns)
    xcenter_index = np.full(shape=m, fill_value=0, dtype=int)

    for i in range(m):
        dmax = 1E10
        for kk in range(k):
            d = 0
            for j in range(n):
                d = d + (df.iloc[:, j][i] - xcenter[kk][j]) ** 2
            d = np.sqrt(d)
            if (d < dmax):
                dmax = d
                xcenter_index[i] = kk

    return xcenter_index

def cluster_kmeans(df,k):

    debug = False

    df_with_index = df.copy()

    m = len(df)
    n = len(df.columns)

    ########### Set the initial value of the centers for each cluster
    xcenter = np.zeros((k, n))

    for j in range(n):
        min = df.iloc[:, j].min()
        max = df.iloc[:, j].max()
        if debug: print(j, min, max)
        for i in range(k):
            xcenter[i][j] = min + (i + 1) / (k + 1) * (max - min)

    if debug: print(xcenter)

    xc_index = xcenter_index(df, xcenter, k)
    df_with_index['Index'] = xc_index
    sse = SSE(df, xcenter, xc_index)

    if debug: print("Initial SSE = ", sse)

    sse_old = sse

    flag = True
    while flag:
        ########## Adjust Centers

        for kk in range(k):
            df_sub = df_with_index[df_with_index['Index'] == kk]
            for j in range(n):
                xcenter[kk][j] = df_sub.iloc[:, j].mean()

        ######## Update index assignment

        xc_index = xcenter_index(df, xcenter,k)
        df_with_index['Index'] = xc_index
        sse = SSE(df, xcenter, xc_index)

        if debug: print("SSE = ", sse)
        if (sse == sse_old):
            flag = False

        sse_old = sse

    xc_index_series = pd.Series(xc_index)

    return xc_index_series, sse


