import numpy as np
import sys
import time
from itertools import combinations

#Function to give the number of occurences of a combination in the transaction list
def give_count_dict(transactions,subsetList):
    accepted = {}
    for subset in subsetList:
        for row in range(len(transactions)):
            if set(subset).issubset(transactions[row]):
                if not str(subset) in accepted:
                    accepted[str(subset)] = 1
                else:
                    accepted[str(subset)] += 1
    return accepted

#Function to give frequent itemsets dictionary
def split_dictionary(initial_dictionary,total_transactions,minimum_support):
    accepted_dictionary = {}
    #rejected_dictionary = {}
    accepted_list = []
    for key in initial_dictionary:
        if (initial_dictionary[key]/float(total_transactions)) >= (minimum_support/100.0):
            #adding frequent itemsets to dictionary and list            
            accepted_list.insert(0,key[1:-1].replace("'",'').split(', '))
            accepted_dictionary[key] = initial_dictionary[key]
    return accepted_dictionary, accepted_list

#Function to return combinations based on the accepted list of frequent itemsets
def getCombinations(accepted_list, length):
    combinations = list()
    
    for i in range(len(accepted_list)):
        for j in range(i+1, len(accepted_list)):
            if(accepted_list[i][:length-2] == accepted_list[j][:length-2]):
                combinations.insert(0,sorted(set(accepted_list[i]).union(set(accepted_list[j]))))
            
    return combinations

filename = raw_input("Enter filename: ")
data = [line.strip().split('\t') for line in open(filename, 'r')]
data = np.asarray(data)
row_cnt = data.shape[0]
attr_cnt = data.shape[1] - 1
disease = data[:,attr_cnt]
transactions = data.astype(str)

support = int(raw_input("Enter minimum support score: "))
start = time.time()
elements_dict = {}
for row in range(len(transactions)):
    for col in range(len(transactions[0])):
        if col != len(transactions[0]):
            transactions[row,col] = 'G'+str(col+1)+'_'+transactions[row,col]
        if not transactions[row,col] in elements_dict:
            elements_dict[transactions[row,col]] = 1
        else:
            elements_dict[transactions[row,col]] += 1

ad, al = split_dictionary(elements_dict,row_cnt,support)

 #To store all the frequent itemsets
all_freq_itemsets = []
al = [[item] for item in list(ad.keys())]
for item in al:
    all_freq_itemsets.append(item)
       
#To store all the frequent itemsets support count
all_freq_itemsets_count = {}
all_freq_itemsets_count.update(ad)
curr_accepted = []

print("Support is set to be "+str(support)+"%")
print("number of length-1 frequent itemsets: "+str(len(al)))

ln = 2
itemset_count = len(al)
while len(al) != 0:    
    curr_accepted = getCombinations(sorted(al),ln)
    count_dict = give_count_dict(transactions,curr_accepted)
    accepted_dict, al = split_dictionary(count_dict,row_cnt,support)
    itemset_count += len(al)
    if len(al) != 0:
        all_freq_itemsets_count.update(accepted_dict)
        all_freq_itemsets = all_freq_itemsets + al
        print("number of length-"+str(ln)+" frequent itemsets: "+str(len(al)))
    ln += 1
print("number of all lengths frequent itemsets: "+str(itemset_count))
#print("Run time = "+str(time.time()-start))

#Returns the support value for the specified itemset
def getSupport(itemset):
    if len(itemset) > 1:
        return all_freq_itemsets_count[str(itemset)]
    else:
        return all_freq_itemsets_count[str(itemset)[2:-2]]
    
#Generates all the rules for the specified itemset with confidence score >= minConfidence 
def getRules(itemset,support_dictionary, minConfidence):
    count = 0
    for l in range(1,len(itemset)):
        for subset in set(combinations(itemset,l)):
            confidence = getSupport(itemset)/float(getSupport(list(subset)))            
            if confidence >= minConfidence/100.0:
                print(str(list(subset))+"-->"+str(list(set(itemset) - set(subset))))#+" = "+str(confidence))
                body.insert(0,list(subset))
                head.insert(0,list(set(itemset) - set(subset)))
                count += 1
    return count

#Returns the indices of rules based on parameter 2 (ANY, NONE or 1) and parameter 3 (itemlist)
def getIndices(rulelist, par2, par3):
    indices = list()
    if par2 == 'ANY':
        for item in par3:
            indices_temp = [i for i, b in enumerate(rulelist) if set([item]).issubset(b)]
            indices.extend(indices_temp)
    elif par2 == 'NONE':
        for i,b in enumerate(rulelist):
            if len(set(par3).intersection(set(b))) == 0:
                indices.insert(0,i)
    elif par2 == '1':
        for i,b in enumerate(rulelist):
            if len(set(par3).intersection(set(b))) == 1:
                indices.insert(0,i)

    return indices

#Fetches the indices of rules based on parameter 1 (BODY, HEAD or RULE)
def template1(par1, par2, par3):
    if par1 == 'BODY':
        indices = getIndices(body, par2, par3)
    elif par1 == 'HEAD':
        indices = getIndices(head, par2, par3)
    else:
        indices1 = getIndices(body, par2, par3)
        indices2 = getIndices(head, par2, par3)
        if par2 == 'NONE':
            indices = list(set(indices1).intersection(indices2))
        elif par2 == 'ANY':
            indices = list(set(indices1).union(indices2))
        elif par2 == '1':
            inter_indices = list(set(indices1).intersection(indices2))
            union_indices = list(set(indices1).union(indices2))
            indices = list(set(union_indices)-set(inter_indices))
    
    return np.unique(indices)

#Fetched the indices of rules based on parameter 1 (BODY, HEAD or RULE) and parameter 2 (length)
def template2(par1, par2):
    if par1 == 'BODY':
        indices = [i for i, b in enumerate(body) if len(b) >= par2]
    elif par1 == 'HEAD':
        indices = [i for i, b in enumerate(head) if len(b) >= par2]
    else:
        indices = [i for i, b in enumerate(head) if len(b)+len(body[i]) >= par2]
    
    return np.unique(indices)

#Prints the rules from the list of indices passed for every template along with the count
def printlist(par_list):
    count = 0
    for item in par_list:
        print(str(body[item]) + "-->" + str(head[item]))
        count += 1
    print("Total count: "+str(count))

ruleCount = 0
head = []
body = []
minConfidence = int(raw_input("Enter minimum confidence score: "))

for item in all_freq_itemsets:
    count = getRules(item, all_freq_itemsets_count, minConfidence)  
    ruleCount += count
    
print("Total number of rules : "+str(ruleCount))

#Accepts inputs from the user for Template 1
def getTemplate1Parameters():
    print("Enter parameters for Template 1")
    par1 = raw_input("Enter first parameter: ").upper()
    if par1 not in ['RULE','BODY','HEAD']:
        print("Invalid first parameter.")
    par2 = raw_input("Enter second parameter: ").upper()
    if par2 not in ['ANY','NONE','1']:
        print("Invalid second parameter.")
    par3 = raw_input("Enter third parameter: ").split()
    return par1, par2, par3

#Accepts inputs from the user for Template 2
def getTemplate2Parameters():
    print("Enter parameters for Template 2")
    par1 = raw_input("Enter first parameter: ").upper()
    if par1 not in ['RULE','BODY','HEAD']:
        print("Invalid first parameter.")
    par2 = int(raw_input("Enter second parameter: "))
    return par1, par2
  
i = 0
#Infinite loop to print association rules for a specified template. Enter 0 as template number to exit.
while i == 0:
    template_id = int(raw_input("Enter template number: "))
    if template_id == 1:
        par1, par2, par3 = getTemplate1Parameters()
        printlist(template1(par1,par2,par3))
    elif template_id == 2:
        par1, par2 = getTemplate2Parameters()
        printlist(template2(par1,par2))
    elif template_id == 3:
        par1 = raw_input("Enter first parameter: ").upper()
        indices = list()
        if len(par1.split("AND")) == 2:
            for template in par1.split("AND"):
                if int(template) == 1:
                    par1, par2, par3 = getTemplate1Parameters()
                    indices.insert(0,template1(par1,par2,par3))
                elif int(template) == 2:
                    par1, par2 = getTemplate2Parameters()
                    indices.insert(0,template2(par1,par2))
                else:
                    print("Invalid combination for Template 3")
                    continue
            printlist(list(set(indices[0]).intersection(set(indices[1]))))
        elif len(par1.split("OR")) == 2:
            for template in par1.split("OR"):
                if int(template) == 1:
                    par1, par2, par3 = getTemplate1Parameters()
                    indices.insert(0,template1(par1,par2,par3))
                elif int(template) == 2:
                    par1, par2 = getTemplate2Parameters()
                    indices.insert(0,template2(par1,par2))
                else:
                    print("Invalid combination for Template 3")
                    continue
            printlist(list(set(indices[0]).union(set(indices[1]))))
    elif template_id == 0:
        break
    else:
        print("else Wrong Template number. Re-enter template numer.")