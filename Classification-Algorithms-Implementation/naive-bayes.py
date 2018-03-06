import numpy as np
import math
import argparse

parser = argparse.ArgumentParser(description='Classification - Naive Bayes.')
parser.add_argument('filename', help='Path to the dataset file.')
                                    
args = parser.parse_args()
filename = args.filename

#Function to check if attribute is String or not (For detecting categorical variables)
def is_string(s):
    try:
        complex(s) # for int, long, float and complex
    except ValueError:
        return True
    return False

#Function to split dataset into train and test - 10 fold cross-validation
def split(test_ind,cross_valid_list):
    test = cross_valid_list[test_ind]
    train = np.vstack([x for i,x in enumerate(cross_valid_list) if i != test_ind])
    return np.asarray(test),np.asarray(train)

#Function to get probabilities of test dataset based on pre-calculated mean and standard deviation 
def get_probabilities(mean, std_dev, test):
    prob = test[:,0:-1] - mean
    prob = np.multiply(prob,prob)
    prob = -1 * prob / (2 * np.multiply(std_dev,std_dev))
    prob = np.exp(prob)
    prob = prob/(math.sqrt(math.pi*2)*std_dev)
    prob = np.prod(prob, axis = 1)
    return prob

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

#Function to exclude categorical
def get_data_excluding_cat(data, indices):
    return np.transpose(np.asarray([data[:,i] for i in range(len(data[0])) if i not in indices]))

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
    
    class_zero = []
    class_one = []
    for j in range(len(train)):
        if int(train[j,-1]) == 1:
            class_one.append(train[j,:])
        elif int(train[j,-1]) == 0:
            class_zero.append(train[j,:])

    class_zero = np.asarray(class_zero)
    class_one = np.asarray(class_one)

    #For categorical
    if len(string_indices) != 0:
        string_prior_prob_one = {}
        string_prior_prob_zero = {}    
        
        for j in string_indices:
            string_prior_prob_one[j] = {}
            string_prior_prob_zero[j] = {}
            for k in np.unique(train[:,j]):
                zero_count = list(class_zero[:,-1].astype(int)).count(0)
                one_count = list(class_one[:,-1].astype(int)).count(1)                
                prior_zero = float(list(class_zero[:,j]).count(k))/zero_count
                prior_one = float(list(class_one[:,j]).count(k))/one_count
                string_prior_prob_zero[j][k] = prior_zero
                string_prior_prob_one[j][k] = prior_one

    #Calculating prior probabilities of zeros and ones
    prior_probablity_zero = float(list(train[:,-1]).count(0))/len(train)
    prior_probablity_one = float(list(train[:,-1]).count(1))/len(train)

    #Excluding categorical
    train_float = get_data_excluding_cat(train, string_indices)

    #Calculating mean of zeros and ones
    mean_zero = np.mean([row[0:-1] for row in train_float if row[-1] == 0],axis = 0)
    mean_one = np.mean([row[0:-1] for row in train_float if row[-1] == 1], axis = 0)

    #Calculating standard deviation of zeros and ones
    std_zero = np.std([row[0:-1] for row in train_float if row[-1] == 0],axis = 0)
    std_one = np.std([row[0:-1] for row in train_float if row[-1] == 1], axis = 0)

    #Calculating probability for categorical explicitly
    string_prob_zero = np.empty(test.shape[0])
    string_prob_one = np.empty(test.shape[0])
    string_prob_one.fill(1.0)
    string_prob_zero.fill(1.0)
    
    if len(string_indices) != 0:
        for t in range(len(test)):
            for i in string_indices:
                string_prob_one[t] *= string_prior_prob_one[i][test[t][i]]
                string_prob_zero[t] *= string_prior_prob_zero[i][test[t][i]]
        
    test_float = get_data_excluding_cat(test, string_indices)

    #Getting the actual probability of zero and one
    prob_zero = prior_probablity_zero * np.multiply(get_probabilities(mean_zero, std_zero, test_float), string_prob_zero)
    prob_one = prior_probablity_one * np.multiply(get_probabilities(mean_one, std_one, test_float) , string_prob_one)

    #Prediciting the class to which test data point belongs to
    test_class = [1 if prob_one[i] > prob_zero[i] else 0 for i in range(len(test))]

    #Getting metrics for calculating precision,accuracy,recall and f-measure
    a, b, c, d = get_metrics(test[:,-1], np.asarray(test_class))
    accuracy += float(a + d)/(a + b + c + d)
    precision += float(a)/(a + c)
    recall += float(a)/(a + b)
    f_measure += float(2 * a) / ((2 * a) + b + c)

print("Accuracy: "+str(accuracy*10))
print("Precision: "+str(precision*10))
print("Recall: "+str(recall*10))
print("F-1 Measure: "+str(f_measure*10))