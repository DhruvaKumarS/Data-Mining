#!/usr/bin/env python
import numpy as np
import sys
from random import randint
from math import sqrt

def eucledian_distance(data,centroid):
    distance = 0
    for i in range(len(data)):
        distance += ((data[i] - centroid[i]) ** 2)

    return sqrt(distance)

def classification(data,centroids):
    euc_all = []
    for i in range(len(centroids)):
        euc_all.insert(i,eucledian_distance(data,centroids[i]))

    return np.argmin(euc_all)+1

def get_data_string(data_point):
    data_str = ''
    for i in range(len(data_point)):
        if len(data_str) == 0:
            data_str = str(data_point[i])
        else:
            data_str = data_str + '\t' + str(data_point[i])
    return data_str


# centroids = np.asarray(
#     [line.strip().split('\t')
    #  for line in open('/Users/sudie/Desktop/CSE-601/CSE601-Project2/centroids.txt', 'r')]).astype(np.float)
centroids = np.asarray(
    [line.strip().split('\t')
     for line in open('centroids', 'r')]).astype(np.float)
centroids = centroids[:,1:len(centroids[0])]
data = [line.strip().split('\t') for line in sys.stdin]
data = np.asarray(data).astype(np.float)

data = np.delete(data,[0,1],axis=1)

k = len(centroids)
clustered_class = []
for i in range(len(data)):
    clustered_class.insert(i, classification(data[i],centroids))
    data_str = get_data_string(data[i])
    # print(cluster)
    print(str(clustered_class[i])+'\t'+data_str)
#print(clustered_class)