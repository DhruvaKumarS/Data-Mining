import numpy as np
import sys
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

#Method to plot clusters and their centroids         
def plot_points(data, clustered_class):
    plt.figure(figsize=[25,12])
    legend = list()
    classes = np.unique(clustered_class)
    colors = cm.Set2(np.linspace(0,1,len(classes)))
    for i in range(len(classes)):
        cluster_data = data[np.where(clustered_class[:]==classes[i])]
        legend.append(plt.scatter(cluster_data[:,0],cluster_data[:,1], c=colors[i], s=10))
    if classes[0] == 0:
        classes[0] = -1
    plt.legend(legend,classes.astype(int))
    plt.title("DBSCAN")
    plt.show()

#Method to append all data points less than eps distance away from the input data point into neighbor_points
def region_query(row1):
    neighbor_points = []
    for row2 in range(len(data)):
        if(eucledian_distance(data[row1],data[row2]) <= eps):
            neighbor_points.append(row2)
    return neighbor_points

#Method to check if the row has at least minpts number of neighbor points
def check_min_points(row,neighbor_points):
    if len(neighbor_points) >= minpts:
        return True
    return False

#Method to check if a data point is already present in a cluster
def check_in_cluster(point):
    for key in all_clusters.keys():
        if point in all_clusters[key]:
            return True
    return False

#Method to add the data point to the specified cluster
def add_to_cluster(cluster_id, row):
    if cluster_id in all_clusters.keys():
        all_clusters[cluster_id].append(row)
    else:
        all_clusters[cluster_id] = []
        all_clusters[cluster_id].append(row)

#Method to expand the cluster of the input data point
def expand_cluster(row,neighbor_points,cluster_id):
    add_to_cluster(cluster_id, row)
    for pt in neighbor_points:
        if pt not in visited:
            visited.append(pt)
            new_neighbor_pts = region_query(pt)
            if(check_min_points(pt,new_neighbor_pts)):
                neighbor_points = list(set(neighbor_points).union(set(new_neighbor_pts)))
                expand_cluster(row,neighbor_points,cluster_id)
                # print(neighbor_points)
        if not check_in_cluster(pt):
            add_to_cluster(cluster_id, pt)
            
filename = raw_input("Enter filename: ")
data = [line.strip().split('\t') for line in open(filename, 'r')]
data = np.asarray(data).astype(np.float)
gene = data[:,0].astype(np.int)
ground_truth = data[:,1].astype(int)

classes = len(np.unique(ground_truth))
data = np.delete(data,[0,1],axis=1)

#eps = 1.03
#minpts = 4
eps = float(raw_input("Enter epsilon value: "))
minpts = int(raw_input("Enter MinPts value: "))

visited = []
noise = []
all_clusters = {}
cluster_id = 0

for row in range(len(data)):
    if row not in visited:    
        visited.append(row)
        neighbor_points = region_query(row)
        if(check_min_points(row,neighbor_points)):
            cluster_id+=1
            expand_cluster(row,neighbor_points,cluster_id)
        else:
            noise.append(row)

clustered_class = np.zeros(ground_truth.shape)
i = 0
for genes in list(all_clusters):
    class_list = np.unique(np.asarray(all_clusters[genes])).astype(int)
    i += len(class_list)
    #print('Cluster '+str(genes)+': '+str(class_list))
    clustered_class[class_list] = genes

ground_truth_matrix = get_incidence_matrix(ground_truth)
cluster_matrix = get_incidence_matrix(clustered_class.astype(int))
m_1_1, m_0_0, m_1_0, m_0_1 = get_count(ground_truth_matrix, cluster_matrix)

print('Jaccard: '+str(m_1_1/(m_1_1 + m_1_0 + m_0_1)))
print('Rand: '+str((m_1_1 + m_0_0)/(m_1_1 + m_0_0 + m_1_0 + m_0_1)))

pca_data = PCA(n_components=2).fit_transform(data)
plot_points(pca_data, clustered_class)