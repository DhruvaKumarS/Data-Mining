k-Means:
-Copy k-means-script.py and the required dataset files into the same directory.
-Run the script as follows:
	python k-means-script.py
-You will be prompted to enter the filename of the dataset as follows:
	Enter filename: <filename of dataset>
-You will be prompted if you want to enter centroids IDs as follows:
	Do you want to enter centroids (Y or N): <Y or N>
-If you entered Y, you will be prompted to enter the centroid indices as follows:
	Enter centroid indices: <comma separated IDs e.g: 3,5,9>
-If you entered N, you will be prompted to enter the value of k and the first k data points will be chosen as centroids.
	Enter value for k: 
-You will be prompted to enter the number of iterations the algorithm is supposed to run for as follows:
	Enter number of iterations: <number of iterations>
-The algorithm runs until it converges i.e: centroids do not change or until the requested number of iterations are completed.
-The Jaccard Coefficient is printed to the console.


Hierarchical Agglomerative clustering with Single Link (Min):
-Copy hierarchical-script.py and the required dataset files into the same directory.
-Run the script as follows:
	python hierarchical-script.py
-You will be prompted to enter the filename of the dataset as follows:
	Enter filename: <filename of dataset>
-You will be prompted to enter the value of k i.e the number of clusters required
	Enter value for k: 
-The algorithm runs until the number of clusters obtained is equal to k
-The Jaccard Coefficient is printed to the console.


Density based (DBSCAN):
-Copy density-based-script.py and the required dataset files into the same directory.
-Run the script as follows:
	python density-based-script.py
-You will be prompted to enter the filename of the dataset as follows:
	Enter filename: <filename of dataset>
-You will be prompted to enter the value of epsilon as follows:
	Enter epsilon value: <floating point value>
-You will be prompted to enter the value of MinPts as follows:
	Enter MinPts value: <integer value>
-The algorithm runs until all the points are either put into a cluster or are marked as noise
-The Jaccard Coefficient is printed to the console.


MapReduce K-Means:
-Make sure hadoop is installed and running.
-Make note of the hadoop home path, hdfs path where files can be placed and the path to the hadoop-streaming-*.jar
-Copy the driver.py, mapper.py, reducer,py, and the dataset to the same directory.
-Run the script as follows:
	python driver.py path_to_hadoop_home path_to_hdfs_directory path_to_hadoop_streaming_jar
-You can also run 
	python driver.py -h to find out about the parameters to be sent.
-You will be prompted to enter the filename of the dataset as follows:
	Enter filename: <filename of dataset>
-You will be prompted if you want to enter centroids IDs as follows:
	Do you want to enter centroids (Y or N): <Y or N>
-If you entered Y, you will be prompted to enter the centroid indices as follows:
	Enter centroid indices: <comma separated IDs e.g: 3,5,9>
-If you entered N, you will be prompted to enter the value of k and the first k data points will be chosen as centroids.
	Enter value for k: 
-You will be prompted to enter the number of iterations the algorithm is supposed to run for as follows:
	Enter number of iterations: <number of iterations>
-The algorithm runs until it converges i.e: centroids do not change or until the requested number of iterations are completed.
-The Jaccard Coefficient is printed to the console.