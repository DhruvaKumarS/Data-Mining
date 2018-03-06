import numpy as np
import sys
from random import randint
from math import sqrt
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans

#Method to return initial centroids. Considering first k values from the input as centroids to get consistent results
def get_initial_centroids():
    centroids = []
    for i in range(k):
        #centroids.append(data[randint(0,data.shape[0]-1)])
        centroids.append(data[i])
        
    return np.asarray(centroids)

#Method to calculate Eucledian distance between a data point and a centroid
def eucledian_distance(data,centroid):
    distance = 0
    for i in range(len(data)):
        distance += ((data[i] - centroid[i]) ** 2)
    
    return sqrt(distance)

#Method to classify a data point to the closest centroid based on Eucledian distance
def classification(data,centroids):
    euc_all = []
    for i in range(len(centroids)):
        euc_all.insert(i,eucledian_distance(data,centroids[i]))
    
    return np.argmin(euc_all)+1

#Method to update centroids by taking mean of all data points of each cluster
def update_centroids(data, clustered_class, old_centroids):
    centroids = []
    for i in range(len(old_centroids)):
        temp_array = np.asarray([np.where(clustered_class[:] == i+1)][0])[0]
        per_cluster = data[temp_array,:]
        if len(per_cluster) == 0:
            centroids.insert(i, old_centroids[i])
        else:
            centroids.insert(i,np.mean(per_cluster, axis=0))
    return np.asarray(centroids)

#Method to check if both data points belong to the same class  
def check_classes(a, b):
    if a == b:
        return 1
    return 0

#Method to populate the incidence matrix for external index calculation    
def get_incidence_matrix(classes):
    incidence_matrix = np.zeros((len(classes),len(classes)))
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
       
#Method to plot clusters and their centroids         
def plot_points(centroids, data, clustered_class):
    plt.figure(figsize=[20,10])
    legend = list()
    classes = np.unique(clustered_class)
    colors = cm.Dark2(np.linspace(0,1,len(classes)))
    for i in range(len(classes)):
        cluster_data = data[np.where(clustered_class[:]==classes[i])]
        legend.append(plt.scatter(cluster_data[:,0],cluster_data[:,1], c=colors[i], alpha=0.75, s=10))
        plt.scatter(centroids[i,0], centroids[i,1], s=130, marker="x", c=colors[i])

    plt.legend(legend,classes)
    plt.title("KMeans")
    plt.show()

filename = raw_input("Enter filename: ")
data = [line.strip().split('\t') for line in open(filename, 'r')]
data = np.asarray(data).astype(np.float)
gene = data[:,0]
ground_truth = data[:,1].astype(int)
classes = len(np.unique(ground_truth))

ground_truth_matrix = get_incidence_matrix(ground_truth)
data = np.delete(data,[0,1],axis=1)

cent = raw_input("Do you want to enter centroids (Y or N): ")
if cent == 'N':
    k = int(raw_input("Enter value for k: "))
    centroids = get_initial_centroids()
else:
    cent_indices = raw_input("Enter centroid indices: ")
    centroids = data[np.asarray(cent_indices.split(',')).astype(int)-1,:].astype(float)

k = len(centroids)
print("K = "+str(k))

iterations = int(raw_input("Enter number of iterations: "))
count = 1
for n in range(iterations):
    clustered_class = []
    for i in range(len(data)):
        clustered_class.insert(i,classification(data[i],centroids))

    clustered_class = np.asarray(clustered_class)

    centroids_new = update_centroids(data, clustered_class, centroids)
    
    if np.array_equal(centroids, centroids_new):
        #print('Equal')
        break
    centroids = centroids_new
    count += 1
print("Iterations: "+str(count))
pca_data = PCA(n_components=2).fit_transform(np.vstack((centroids_new,data)))
pca_centroids = pca_data[0:k,:]
pca_data = pca_data[k:,:]

plot_points(pca_centroids, pca_data, clustered_class)

cluster_matrix = get_incidence_matrix(clustered_class.astype(int))

m_1_1, m_0_0, m_1_0, m_0_1 = get_count(ground_truth_matrix, cluster_matrix)

print('Jaccard: '+str(m_1_1/(m_1_1 + m_1_0 + m_0_1)))
print('Rand: '+str((m_1_1 + m_0_0)/(m_1_1 + m_0_0 + m_1_0 + m_0_1)))
