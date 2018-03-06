import numpy as np
from math import sqrt
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
import argparse

parser = argparse.ArgumentParser(description='Classification - K Nearest Neighbor.')
parser.add_argument('filename', help='Path to the dataset file.')
parser.add_argument('k', help='Value of K.')
                                    
args = parser.parse_args()
filename = args.filename
k = int(args.k)

#Method to calculate Eucledian distance between a data point and a centroid
def eucledian_distance(data_point,test_point):
    distance = 0
    for i in range(len(test_point)):
        if i in string_indices:
            distance += 1.0 if test_point[i] != data_point[i] else 0.0
        else:
            distance += ((data_point[i] - test_point[i]) ** 2)
    return sqrt(distance)

#Method to classify a test data point
def classify(train_data, test_point):
    distance_list = []
    for train in train_data:
        distance_list.append(eucledian_distance(train[0:-1],test_point[0:-1]))
    k_smallest = np.argsort(np.asarray(distance_list))[:k]
    return retrieve_classification(k_smallest,train_data[:,-1])

#Method to retrieve the classification using the k nearest train data points
def retrieve_classification(k_smallest,train_last_column):
    one = 0
    zero = 0
    for x in k_smallest:
        if train_last_column[x] == 0:
            zero +=1
        elif train_last_column[x] == 1:
            one += 1
    return 1 if max(one,zero) == one else 0

#Method to split the data for cross-validation
def split(test_ind,cross_valid_list):
    test = cross_valid_list[test_ind]
    train = np.vstack([x for i,x in enumerate(cross_valid_list) if i != test_ind])
    return np.asarray(test),np.asarray(train)

#Method to check if the attribute value is categorical(string) or continuous
def is_string(s):
    try:
        complex(s)
    except ValueError:
        return True
    return False

#Method to retrieve true positive, false positive, false negative and true negative for the classification
def get_metrics(actual, predicted):
    a = b = c = d = 0
    for i in range(len(actual)):
        if actual[i] == 1 and predicted[i] == 1:
            a += 1
        elif actual[i] == 1 and predicted[i] == 0:
            b += 1
        elif actual[i] == 0 and predicted[i] == 1:
            c += 1
        elif actual[i] == 0 and predicted[i] == 0:
            d += 1
    return a, b, c, d
#Method to normalize the data using mean and standard deviation
def get_normalized_data(train_data, test_data_point):
    temp = np.vstack((train_data[:,0:-1],test_data_point[0:-1].reshape((1,test_data_point[0:-1].shape[0]))))

    normalized_data = (temp - np.mean(temp,axis = 0))/np.std(temp,axis = 0)
    normalized_train = np.hstack((normalized_data[0:-1,:],train_data[:,-1].reshape(train_data[:,-1].shape[0],1)))
    normalized_test = np.hstack((normalized_data[-1,:],test_data_point[-1]))
    return normalized_train, normalized_test
    

# filename = raw_input("Enter filename: ")
data = [line.strip().split('\t') for line in open(filename, 'r')]
data = np.asarray(data)
output_class = data[:,-1].reshape((data.shape[0],1)).astype(int)
data = data[:,0:-1]

string_indices = []
for i in range(len(data[0])):
    if is_string(data[0][i]):
        string_indices.append(i)
for i in string_indices:
    unique_strings = np.unique(data[:,i])
    replacement_vals = list(range(len(unique_strings)))
    dictionary = dict(zip(unique_strings, replacement_vals))
    for j in range(len(data[:,i])):
        data[j][i] = dictionary.get(data[j][i])

data = data.astype(float)
data = np.append(data,output_class,axis=1)

cross_valid = np.array_split(data, 10)
accuracy = precision = recall = f_measure = 0


for i in range(len(cross_valid)):
    test, train = split(i,cross_valid)
    test_class = []
    for row in range(len(test)):
        normalized_train, normalized_test = get_normalized_data(train, test[row])
        test_class.append(classify(normalized_train, normalized_test))
    a, b, c, d = get_metrics(test[:,-1], np.asarray(test_class))
    accuracy += (float(a + d)/(a + b + c + d))
    precision += (float(a)/(a + c))
    recall += (float(a)/(a + b))
    f_measure += (float(2 * a) / ((2 * a) + b + c))
print("Accuracy: "+str(accuracy * 10))
print("Precision: "+str(precision * 10))
print("Recall: "+str(recall * 10))
print("F-1 Measure: "+str(f_measure * 10))