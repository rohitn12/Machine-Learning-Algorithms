import scipy.optimize , scipy.io
import sys
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

Path = #add path



data1 = scipy.io.loadmat(Path + "\ex7data2")
x1 = data1['X']
m = x1.shape[1]
print(m)
k = 3 #no of clusters


def randomIntializeCentroid(x , k):
    
    return np.random.randint(np.amax(x) , size =  (k ,x.shape[1]))

init_centroid = randomIntializeCentroid(x1 , K)

##finding the closest centroid to each point in the dataset 

def closestCentroid(x ,centroid):
    m = len(x1)
    k = centroid.shape[0]
    closest_centroids = np.zeros((m))
    
    for i in range(m):
        min_distance = 100000
        for j in range(k):
            distance = np.sum((x[i,:] - centroid[j,:])**2)
            if distance < min_distance:
                min_distance = distance
                closest_centroids[i] = j
    return closest_centroids

def compute_centroid(x , closest_centroids , k):
    m = x.shape[0]
    n = x.shape[1]
    centroids = np.zeros((k,n))
    for i in range(k):
        indices = np.where(i == closest_centroids)
        centroids[i,:] = (np.sum(x[indices,:] , axis =1) / len(indices[0]))
    
    return centroids

def kmeans(x , init_centroid , max_iter):
    m = x.shape[0]
    n = x.shape[1]
    centroid = init_centroid
    idx = np.zeros(m)
    k = init_centroid.shape[0]
    for i in range(max_iter):
        idx = closestCentroid(x , centroid)
        centroid = compute_centroid(x , idx , k)
    return idx , centroid

