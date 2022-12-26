from pyspark.mllib.clustering import KMeans
from numpy import array, random
from math import sqrt
from pyspark import SparkConf, SparkContext
from sklearn.preprocessing import scale

K = 5

# Boilerplate Spark stuff:
conf = SparkConf().setMaster("local").setAppName("SparkKMeans")
sc = SparkContext(conf = conf)

#Create fake income/age clusters for N people in k clusters
def createClusteredData(N, k):
    random.seed(10)
    pointsPerCluster = float(N)/k
    X = []
    for i in range (k):
        incomeCentroid = random.uniform(20000.0, 200000.0)
        ageCentroid = random.uniform(20.0, 70.0)
        for j in range(int(pointsPerCluster)):
            X.append([random.normal(incomeCentroid, 10000.0), random.normal(ageCentroid, 2.0)])
    X = array(X)
    return X

random.seed(0)

# Load the data; note I am normalizing it with scale() - very important!
data = sc.parallelize(scale(createClusteredData(100, K)))

# Build the model (cluster the data)
clusters = KMeans.train(data, K, maxIterations=10,
        initializationMode="random")

# Print out the cluster assignments
resultRDD = data.map(lambda point: clusters.predict(point)).cache()

print("Counts by value:")
counts = resultRDD.countByValue()
print(counts)

print("Cluster assignments:")
results = resultRDD.collect()
print(results)


# Evaluate clustering by computing Within Set Sum of Squared Errors
def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

WSSSE = data.map(lambda point: error(point)).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = " + str(WSSSE))

# Things to try:
# What happens to WSSSE as you increase or decrease K? Why?
K = 3
data = sc.parallelize(scale(createClusteredData(100, K)))
clusters = KMeans.train(data, K, maxIterations=10,
        initializationMode="random")
resultRDD = data.map(lambda point: clusters.predict(point)).cache()
WSSSE = data.map(lambda point: error(point)).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = " + str(WSSSE))

K = 7
data = sc.parallelize(scale(createClusteredData(100, K)))
clusters = KMeans.train(data, K, maxIterations=10,
        initializationMode="random")
resultRDD = data.map(lambda point: clusters.predict(point)).cache()
WSSSE = data.map(lambda point: error(point)).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = " + str(WSSSE))

# We can see that when the K is decreased to 3, the WSSSE increases to 26. We
# can also see that when the K is increased to 7, the WSSSE decreases to 20

# What happens if you don't normalize the input data before clustering?
K = 5
data = sc.parallelize(createClusteredData(100, K))
clusters = KMeans.train(data, K, maxIterations=10,
        initializationMode="random")
resultRDD = data.map(lambda point: clusters.predict(point)).cache()
WSSSE = data.map(lambda point: error(point)).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = " + str(WSSSE))

# We can see that without scaling, the WSSE increases dramatically to
# 608008. This highlights the importance of scaling data before inputting it.

# What happens if you change the maxIterations parameter? To do this let's
# test a maxIterations of 5 and one of 50
K = 5
data = sc.parallelize(scale(createClusteredData(100, K)))
clusters = KMeans.train(data, K, maxIterations=5,
        initializationMode="random")
resultRDD = data.map(lambda point: clusters.predict(point)).cache()
WSSSE = data.map(lambda point: error(point)).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = " + str(WSSSE))

K = 5
data = sc.parallelize(scale(createClusteredData(100, K)))
clusters = KMeans.train(data, K, maxIterations=50,
        initializationMode="random")
resultRDD = data.map(lambda point: clusters.predict(point)).cache()
WSSSE = data.map(lambda point: error(point)).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = " + str(WSSSE))

# We can see that decreasing the parameter value increases the WSSE to 34.
# Increasing the parameter value does not change the WSSE much in this case

# What happens if you change initializationMode to "k-means||"
K = 5
data = sc.parallelize(scale(createClusteredData(100, K)))
clusters = KMeans.train(data, K, maxIterations=5,
        initializationMode="k-means||")
resultRDD = data.map(lambda point: clusters.predict(point)).cache()
WSSSE = data.map(lambda point: error(point)).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = " + str(WSSSE))

# We can see that we get a lower WSSE value of 19.97. The previous parameter
# value of 'random' chose a random initial cluster whereas the 'k-means||' value
# used a parallel variant of k-means++