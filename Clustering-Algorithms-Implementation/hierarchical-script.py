import numpy as np
import sys
import time
from random import randint
from math import sqrt
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm


#Method to calculate Eucledian distance between two clusters
def eucledian_distance(data,centroid):
    distance = 0
    for i in range(len(data)):
        distance += ((data[i] - centroid[i]) ** 2)
    
    return float(sqrt(distance))

#Method to check if both data points belong to the same class
def check_classes(a, b):
    if a == b:
        return 1
    return 0

#Method to populate the incidence matrix for external index calculation  
def get_incidence_matrix(classes):
    incidence_matrix = np.zeros((len(classes),len(classes)))
    #print(incidence_matrix.shape)
    for i in range(len(classes)):
        for j in range(len(classes)):
            val =check_classes(classes[i], classes[j])
            incidence_matrix[i][j] = val
    return incidence_matrix

#Method to get the parameters needed from the indicence matrix to calculate external index
def get_count(P, C):
    m_1_1 = m_0_0 = m_1_0 = m_0_1 = 0.0
    for i in range(len(P)):
        for j in range(len(P)):
            if P[i][j] == C[i][j]:
                if P[i][j] == 1:
                    m_1_1 += 1
                else:
                    m_0_0 += 1
            else:
                if P[i][j] == 1:
                    m_1_0 += 1
                else:
                    m_0_1 += 1
                
    return m_1_1, m_0_0, m_1_0, m_0_1

#Method to return the index of minimum value from the dictionary representation of distance matrix
def nested_dict_min(dictionary):
    m = float('inf')
    row = 0
    col = 0
    for subdict in dictionary.keys():
        for subsub in dictionary[subdict].keys():
            value = dictionary[subdict][subsub]
            if value < m:
                m = value
                row = subdict
                col = subsub
    return row,col

#Method to populate the dictionary representation of the distance matrix
def populate_all_distances(all_distances):
    for i in range(len(data)):
        all_distances[str(i)] = {}
        for j in range(len(data)):
            all_distances[str(i)][str(j)] = {}
            if i != j:
                value = eucledian_distance(data[i],data[j])
                all_distances[str(i)][str(j)] = value
            else:
                all_distances[str(i)][str(j)] = float('inf')
    return all_distances

#Method to merge two clusters and modify the distance matrix
def modify_alldistances(row,col,all_distances):
    new_key = row+','+col
    key_list = all_distances.keys()
    all_distances[new_key] = {}
    for i in key_list:
        all_distances[new_key][i] = min(all_distances[row][i],all_distances[col][i])
        all_distances[i][new_key] = min(all_distances[row][i],all_distances[col][i])
    all_distances[new_key][new_key] = float('inf')
    del all_distances[row]
    del all_distances[col]
    for i in list(all_distances):
        del all_distances[i][row]
        del all_distances[i][col]
    return all_distances

#Method to plot clusters        
def plot_points(data, clustered_class):
    plt.figure(figsize=[25,12])
    legend = list()
    classes = np.unique(clustered_class)
    colors = cm.Set2(np.linspace(0,1,len(classes)))
    for i in range(len(classes)):
        cluster_data = data[np.where(clustered_class[:]==classes[i])]
        legend.append(plt.scatter(cluster_data[:,0],cluster_data[:,1], c=colors[i], s=10, alpha=0.9))

    plt.legend(legend,classes.astype(int))
    plt.title("Hierarchical - Min")
    plt.show()

filename = raw_input("Enter filename: ")
start = time.time()
data = [line.strip().split('\t') for line in open(filename, 'r')]
data = np.asarray(data).astype(np.float)
gene = data[:,0].astype(np.int)
ground_truth = data[:,1].astype(int)

ground_truth_matrix = get_incidence_matrix(ground_truth)

classes = len(np.unique(ground_truth))
k = int(raw_input("Enter value for k: "))
print("K = "+str(k))
data = np.delete(data,[0,1],axis=1)

all_distances = {}
all_distances = populate_all_distances(all_distances)

i = 1
while(len(list(all_distances)) != k):
    row,col = nested_dict_min(all_distances)
    all_distances = modify_alldistances(row,col,all_distances)

i = 1
clustered_class = np.zeros(ground_truth.shape)

for genes in list(all_distances):
    class_list = np.asarray(genes.split(',')).astype(int)
    clustered_class[class_list] = i
    i += 1

cluster_matrix = get_incidence_matrix(clustered_class.astype(int))
m_1_1, m_0_0, m_1_0, m_0_1 = get_count(ground_truth_matrix, cluster_matrix)

print('Jaccard: '+str(m_1_1/(m_1_1 + m_1_0 + m_0_1)))
print('Rand: '+str((m_1_1 + m_0_0)/(m_1_1 + m_0_0 + m_1_0 + m_0_1)))
#print('Time taken: '+str(time.time()-start))

pca_data = PCA(n_components=2).fit_transform(data)
plot_points(pca_data, clustered_class)