import pandas as pd
import numpy as np



def get_sse(obs, centroids, labels):
    """
    Finds the sum of the square error (distance) between
    centroids and its labeled observations

    Arguments:
        obs  numpy array of observations (m obserations x n attributes)
        centroids  k centroids of m dimensions
        labels  m labels; one for each observation
     """

    m = obs.shape[0]
    n = obs.shape[1]
    sse = 0

    for i in range(m):
        for j in range(n):
            sse = sse + (obs[i][j] - centroids[labels[i]][j]) ** 2

    return sse
    pass




def find_labels(obs, centroids):
    """
    Labels each observation in df based upon
    the nearest centroid. 
    
    Args:
        obs  numpy array of observations (m obserations x n attributes)
        centroids  k centroids of m dimensions
    
    Returns:
        a numpy array of labels, one for each observation
    """

    debug = False

    m = obs.shape[0]
    n = obs.shape[1]
    k = len(centroids)
    labels = np.full(shape=m, fill_value=0, dtype=int)
    if debug: print ("In labels: ",m,n,k)

    for i in range(m):
        dmax = 1E10
        for kk in range(k):
            d = 0
            for j in range(n):
                d = d + (obs[i][j] - centroids[kk][j])**2
            d = np.sqrt(d)
            if (d < dmax):
                dmax = d
                labels[i] = kk

    return labels
    pass
    



def recompute_centroids(obs, centroids, labels):
    """
    Find the new location of the centroids by
    finding the mean location of all points assigned
    to each centroid
    
    Arguments:
        obs  numpy array of observations (m obserations x n attributes)
        centroids  k centroids of m dimensions
        labels  m labels; one for each observation
    
    Returns:
        None; the centroids data frame is updated
    """
    debug = False

    obs_with_index = pd.DataFrame(obs)

    m = len(obs_with_index)
    n = len(obs_with_index.columns)
    k = len(centroids)
    if debug: print ('In recompute_centroids: ',m,n,k)

    new_centroids = np.zeros((k,n))

    if k < 1:  # There are no clusters, so I'm returning None
        return None

    # Add this index to the dataframe.
    obs_with_index['Index'] = labels

    ########## Adjust Centers

    for kk in range(k):
        obs_sub = obs_with_index[obs_with_index['Index'] == kk]
        if debug: print (obs_sub)
        if (len(obs_sub)==0):
            new_centroids = centroids.copy()
        else:
            for j in range(n):
                new_centroids[kk][j] = float(obs_sub.mean(axis=0)[j])
                if debug: print ('kk, centroid: ',kk,new_centroids[kk][j])

    return new_centroids
    pass

def cluster_kmeans(obs, k):
    """
    Clusters the m observations of n attributes 
    in the Pandas' dataframe df into k clusters.
    
    Euclidean distance is used as the proximity metric.
    
    Arguments:
        obs  numpy array of observations (m obserations x n attributes)
        k    the number of clusters to search for
        
    Returns:
        a m-sized numpy array of the cluster labels
        
        the final Sum-of-Error-Squared (SSE) from the clustering

        a k x n numpy array of the centroid locations
    """

    if k < 1:  # There are no clusters, so I'm returning None
        return None

    debug = False

    obs_with_index = pd.DataFrame(obs)

    m = len(obs_with_index)
    n = len(obs_with_index.columns)

    centroids = np.zeros((k, n))

    # choose the initial positions of the centroids RANDOMLY
    for j in range(n):
        min = obs_with_index.iloc[:, j].min()
        max = obs_with_index.iloc[:, j].max()
        if debug: print(j, min, max)
        for i in range(k):
            centroids[i][j] = np.random.choice([min,max])

    if debug: print(centroids)

    # For the chosen center positions, create the center index array.
    labels = find_labels(obs,centroids)
    if debug: print (labels)

    # Get the starting SSE value for the initial center positions.
    sse = get_sse(obs, centroids, labels)

    if debug: print("Initial SSE = ", sse)

    sse_old = sse
    flag = True
    while flag:
        ########## Adjust Centers

        centroids = recompute_centroids(obs,centroids,labels)

        ######## Update index assignment
        if debug: print ('Updating labels ... ')
        labels = find_labels(obs, centroids)
        if debug: print (labels)
        obs_with_index['Index'] = labels
        sse = get_sse(obs, centroids, labels)
        if debug: print("SSE = ", sse)

        if (sse == sse_old):
            if debug: print('Final Centers: ', centroids)
            flag = False

        sse_old = sse

    return labels,sse,centroids
    pass
