from pyspark import SparkContext
sc = SparkContext()

# Show RDD
sc.textFile('example2.txt')

# Save a reference to this RDD
text_rdd = sc.textFile('example2.txt')

# Map a function (or lambda expression) to each line
# Then collect the results.
text_rdd.map(lambda line: line.split()).collect()

# Collect everything as a single flat map
text_rdd.flatMap(lambda line: line.split()).collect()

services = sc.textFile('services.txt')
services.take(2)

#total sales per state
cleanServ = services.map(lambda x: x[1:] if x[0]=='#' else x).map(lambda x: x.split())
cleanServ.collect()

#convert numbers to float not string
cleanServ.map(lambda lst: (lst[3],lst[-1]))\
         .reduceByKey(lambda amt1,amt2 : float(amt1)+float(amt2))\
         .collect()

# Grab state and amounts
# Add them
# Get rid of ('State','Amount')
# Sort them by the amount value
cleanServ.map(lambda lst: (lst[3],lst[-1]))\
.reduceByKey(lambda amt1,amt2 : float(amt1)+float(amt2))\
.filter(lambda x: not x[0]=='State')\
.sortBy(lambda stateAmount: stateAmount[1], ascending=False)\
.collect()

    
