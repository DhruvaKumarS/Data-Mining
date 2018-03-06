import numpy as np
import sys
import argparse

parser = argparse.ArgumentParser(description='Classification - Decision Tree.')
parser.add_argument('filename', help='Path to the dataset file.')
                                    
args = parser.parse_args()
filename = args.filename

#Function for calculating gini_index
def get_gini_index(left, right):
    total_count = len(left) + len(right)
    count_left_one = count_left_zero = count_right_one = count_right_zero = 0
    if len(left) != 0:
        count_left_one = float(list(left[:,-1]).count(1)) /len(left)
        count_left_zero = float(list(left[:,-1]).count(0)) / len(left)
    if len(right) != 0:
        count_right_one = float(list(right[:,-1]).count(1)) / len(right)
        count_right_zero = float(list(right[:,-1]).count(0)) / len(right)
    
    gini_index_left = 1.0 - ((count_left_one * count_left_one) + (count_left_zero * count_left_zero))
    gini_index_right = 1.0 - ((count_right_one * count_right_one) + (count_right_zero * count_right_zero))
    
    gini_index = ((gini_index_left * len(left)) + (gini_index_right * len(right))) / total_count
    return gini_index

#Function to split the data into two sublists (left and right)
def get_split_data(data, split_value, col):
    left = []
    right = []
    for row in range(len(data)):
        if col not in string_indices:
            if data[row][col] <= split_value:
                left.append(data[row])
            else:
                right.append(data[row])
        else:
            if data[row][col] == split_value:
                left.append(data[row])
            else:
                right.append(data[row])
    return np.asarray(left), np.asarray(right)

#Function for returning the split-node in the decision tree
def get_node(data):
    min_error = float('inf')
    node = {}
    for col in range(len(data[0]) - 1):
        for row in range(len(data)):
            left, right = get_split_data(data, data[row][col], col)
            error = get_gini_index(left, right)
            if error < min_error:
                min_error = error
                node['split_index'] = row
                node['split_value'] = data[row][col]
                node['split_attr'] = col
                node['left'] = left
                node['right'] = right
    return node

#Function to get the classification at that node
def get_class(left, right):
    zero_count = 0
    one_count = 0;
    if len(left) != 0:
        zero_count += list(left[:,-1]).count(0)
        one_count += list(left[:,-1]).count(1)
    if len(right) != 0:
        zero_count += list(right[:,-1]).count(0)
        one_count += list(right[:,-1]).count(1)
    return 1 if one_count > zero_count else 0

#Function to check if everything belongs to the same class
def check_class(data):
    if len(np.unique(data[:,-1])) == 1:
        return True
    return False

#Funciton to build the decision tree
def get_decision_tree(node, depth):
    left = node['left']
    right = node['right']
    del(node['left'])
    del(node['right'])

    if len(left) == 0 or len(right) == 0:
        node['left'] = node['right'] = get_class(left, right)
        return node 

    if len(left) > 0:
        if check_class(left):
            node['left'] = get_class(left, list())
        else:
            node['left'] = get_decision_tree(get_node(left), depth + 1)
    if len(right) > 0:
        if check_class(right):
            node['right'] = get_class(list(), right)
        else:
            node['right'] = get_decision_tree(get_node(right), depth + 1)
    return node

#Function to classify the test record using the decision tree
def classify(node, test_data_point):
    if node == 0 or node == 1:
        return node
    elif node['split_attr'] not in string_indices:
        if test_data_point[node['split_attr']] < node['split_value']:
            if node['left'] == 0 or node['left'] == 1:
                return node['left']                                 
            else:
                return classify(node['left'], test_data_point)
        else:
            if node['right'] == 0 or node['right'] == 1:
                return node['right']
            else:
                return classify(node['right'], test_data_point)
    else:
        if test_data_point[node['split_attr']] == node['split_value']:
            if node['left'] == 0 or node['left'] == 1:
                return node['left']                                 
            else:
                return classify(node['left'], test_data_point)
        else:
            if node['right'] == 0 or node['right'] == 1:
                return node['right']
            else:
                return classify(node['right'], test_data_point)

#Function to split dataset into train and test - 10 fold cross-validation
def split(test_ind,cross_valid_list):
    test = cross_valid_list[test_ind]
    train = np.vstack([x for i,x in enumerate(cross_valid_list) if i != test_ind])
    return np.asarray(test),np.asarray(train)

#Function to check if attribute is String or not (For detecting categorical variables)
def is_string(s):
    try:
        complex(s) # for int, long, float and complex
    except ValueError:
        return True

    return False

#Function to get metrics - TruePositive, FalsePositive, TrueNegative, FalseNegative for calculating accuracy/precision/recall/f-measure
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

#Opening file handle and retrieving content after removing last column (Output class)
data = [line.strip().split('\t') for line in open(filename, 'r')]
data = np.asarray(data)#.astype(np.float)
output_class = data[:,-1].reshape((data.shape[0],1)).astype(int)
data = data[:,0:-1]

#Detecting categorical variables in the dataset and giving them numerical values
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

#Appending the output class at the last after converting the dataset as np array of type float
data = data.astype(float)
data = np.append(data,output_class,axis=1)
cross_valid = np.array_split(data, 10)
accuracy = precision = recall = f_measure = 0

#Start of actual algorithm 
for i in range(len(cross_valid)):
    test, train = split(i,cross_valid)
    test_class = []
    root = get_node(train)
    root = get_decision_tree(root, 1)
    for row in range(len(test)):
        test_class.append(classify(root,test[row]))
    a, b, c, d = get_metrics(test[:,-1], np.asarray(test_class))
    accuracy += float(a + d)/(a + b + c + d)
    precision += float(a)/(a + c)
    recall += float(a)/(a + b)
    f_measure += float(2 * a) / ((2 * a) + b + c)
print("Accuracy: "+str(accuracy*10))
print("Precision: "+str(precision*10))
print("Recall: "+str(recall*10))
print("F-1 Measure: "+str(f_measure*10))