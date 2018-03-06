#!/usr/bin/env python
import numpy as np
import sys
from random import randint
from math import sqrt

def get_data_string(data_point):
    data_str = ''
    for i in range(len(data_point)):
        if len(data_str) == 0:
            data_str = str(data_point[i])
        else:
            data_str = data_str + '\t' + str(data_point[i])
    return data_str

def update_centroids(data, clustered_class, k):
    centroids = []
    for i in range(k):
        temp_array = np.asarray([np.where(clustered_class[:] == i+1)][0])[0]
        per_cluster = data[temp_array,:]
        #print(per_cluster.shape)
        centroids.insert(i,np.mean(per_cluster, axis=0))
        #print("Centroid "+str(i+1))
        #print(centroids[i])
    return np.asarray(centroids) 

data = np.asarray([line.strip().split('\t') for line in sys.stdin])
#print(data)
cluster = data[:,0].astype(np.int)
data = data[:,1:len(data[0])].astype(np.float)
cluster_count = len(np.unique(cluster))
centroids_new = update_centroids(data, cluster, len(np.unique(cluster)))

for i in range(cluster_count):
    print(str(i+1)+'\t'+get_data_string(centroids_new[i]))