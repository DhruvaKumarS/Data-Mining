import numpy as np
import sys
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE

def plotPoints(transform,classes,disease,title):
    legend = list()
    plt.figure(figsize=[40,20])
    colors = cm.Set1(np.linspace(0,1,len(classes)))
    for i in range(len(classes)):
        transform_class = transform[np.where(disease==classes[i])]
        legend.append(plt.scatter(transform_class[:,0],transform_class[:,1], c=colors[i], alpha=0.75))
    plt.legend(legend,classes)
    plt.title(title)
    plt.show()

def pca():
    mean = np.mean(attr, axis=0)
    adjusted_attr = attr - mean
    cov_mat = np.dot(np.transpose(adjusted_attr), adjusted_attr)/row_cnt
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    max_indices = eig_vals.argsort()[-2:][::-1]
    max_eig_vecs = eig_vecs[:,max_indices]
    max_eig_vecs = max_eig_vecs.T.reshape((max_eig_vecs.shape[1],max_eig_vecs.shape[0]))
    transform = np.dot(adjusted_attr, max_eig_vecs.T)
    return transform

filename = raw_input("Enter filename: ")
data = [line.strip().split('\t') for line in open(filename, 'r')]
data = np.asarray(data)
row_cnt = data.shape[0]
attr_cnt = data.shape[1] - 1
disease = data[:,attr_cnt]
attr = np.delete(data, attr_cnt, axis=1).astype(np.float)
classes = np.unique(disease)

transform = pca()
plotPoints(transform,classes,disease,'PCA for dataset '+str(filename))

#SVD part
U, S, V = np.linalg.svd(attr, full_matrices=False)
transform = np.dot(U, np.diag(S))#[:,[0,1]]
plotPoints(transform,classes,disease,'SVD for dataset '+str(filename))

#t-SNE part
transform = TSNE(n_components=2).fit_transform(attr)
plotPoints(transform,classes,disease,'t-SNE for dataset '+str(filename))