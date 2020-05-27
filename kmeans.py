import numpy as np
import pandas as pd
from PIL import Image

grey = np.array(Image.open('north-africa-1940s-grey.png'))
grey = grey.reshape(-1).reshape(-1, 1).astype(float)

def kmeans(X:np.ndarray, k:int, centroids=None, tolerance=1e-2):
    if centroids == 'kmeans++':
        centroid = np.zeros(shape=(k,X.shape[1]))
        initial_centroid = X[np.random.randint(X.shape[0])]
        centroid[0] = initial_centroid
        for i in range(1, k):            
            distance_list = [((X - centroid[index])**2).sum(axis=1).reshape(-1,1) for index in range(i)]
            distance_array = np.concatenate(distance_list, axis=1)
            temp = np.argmin(distance_array, axis =1)
            index_dict = {distance_array[i][j]:i for i,j in enumerate(temp)}
            max_index = index_dict[max(index_dict.keys())]
            centroid[i] = X[max_index]
    else:
        centroid = X[np.random.randint(X.shape[0], size=k)]
    while tolerance:
        cluster_list = [[] for i in range(k)]
        distance_list = [((X - centroid[i])**2).sum(axis=1).reshape(-1, 1) for i in range(len(centroid))]
        distance_array = np.concatenate(distance_list, axis = 1)
        temp = list(np.argmin(distance_array, axis =1))
        for i, j in enumerate(temp):
            cluster_list[j].append(i)            
        for i in range(len(cluster_list)):
            if len(cluster_list[i]) == 0: 
                cluster_list[i].append(np.random.randint(X.shape[0]))
        centroid_new = np.array([sum(X[cluster])/len(X[cluster]) for cluster in cluster_list])
        if ((centroid_new - centroid)**2).sum()/len(centroid_new) < tolerance:
            tolerance = False
        centroid = centroid_new 
    return centroid, cluster_list

def reassign_colors(X, centroid, cluster_list):
    for i in range(len(centroid)):
        X[cluster_list[i]] = centroid[i]
    return X.astype(np.uint8)

def reassign_grey(X, centroid, cluster_list):
    X_ = X.reshape(-1,1).astype(float)
    for i in range(len(centroid)):
        X_[cluster_list[i]] = centroid[i]
    return X_.astype(np.uint8)




