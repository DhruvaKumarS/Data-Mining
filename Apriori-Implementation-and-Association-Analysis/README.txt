Part 2 - Apriori and Association rules
-Make sure the apriori_script.py and the dataset file are in the same directory
-Run the file using python2 as follows:
	python2 apriori_script.py
-You will be prompted to enter the filename as follows:
	Enter filename: <Type the filename>
	e.g. Enter filename: associationruletestdata.txt
-You will be prompted to enter the minimum support score as follows:
	Enter minimum support score: <Enter the minimum support %>
	e.g. Enter minimum support score: 50
-The itemsets count will be displayed as follows:
	Support is set to be 50%
	number of length-1 frequent itemsets: 109
	number of length-2 frequent itemsets: 63
	number of length-3 frequent itemsets: 2
-You will be prompted to enter the minimum confidence score as follows:
	Enter minimum confidence score: <Enter the minimum confidence %>
	e.g. Enter minimum confidence score: 70
-All the association rules with confidence score >= the minimum confidence score entered will be displayed along with the count.
-You will be prompted to enter the template number as follows:
	Enter template number: <1, 2 or 3>
-You will be prompted to enter the parameters based on the template
-Template 1 has 3 parameters:	First parameter: <RULE, BODY or HEAD>
				Second parameter: <ANY, NONE or 1>
				Third parameter: <List of space separated items>
-Template 2 has 2 parameters:	First parameter: <RULE, BODY or HEAD>
				Second parameter: <Number>
-Template 3 is a combination of Template 1 and Template 2 using AND or OR.
				First parameter: <Combination of template 1 and 2>
				e.g. 1or2, 1and2, 1and1, etc.
-Based on the combination entered, you will prompted to enter parameters for the respective template number.
-The rules which satisfy the entered template number and the parameters will be printed along with the count.