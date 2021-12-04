import json
from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import Row, SQLContext, SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import * 

def json_data(rdd_data):
  data2 = {}
  for i in rdd_data:
    data2 = json.loads(i, strict = False)
  return data2

def list_to_tuple(list1):
  return tuple(list1)

def process1(rdd,count):
	if not rdd.isEmpty():
	#PARSING THE JSON FILE AND EXTRACTING ITS ROWS
		rows1 = []
		rdd_data = rdd.collect()
		data1 = json_data(rdd_data)
		for i in data1.keys():
			x = list()
			x.append(len(str(data1[i]['feature1'])))
			x.append(str(data1[i]['feature0']).strip(' '))
			x.append(str(data1[i]['feature1']).strip(' '))
			x.append(str(data1[i]['feature2']).strip(' '))
    
			rows1.append(list_to_tuple(x))
		print("RECIEVED BATCH OF LENGTH ", len(rows1))
		#print(rows1)
		rdd2 = sc.parallelize(rows1)
		#SVGD, BERNOULLI AND MULTINOMIAL CLASSIFIERS TO BE USED
		print("COMPLETED \n \n")
            
conf = SparkConf()
conf.setAppName("BD")
sc = SparkContext(conf=conf)
ssc = StreamingContext(sc, 2)
sqlc = SQLContext(sc)
spark = SparkSession(sc)


dataStream = ssc.socketTextStream("localhost",6100)
#dataStream.pprint()
lines = dataStream.flatMap(lambda line: line.split("\n"))
count = 1
lines.foreachRDD(lambda rdd : process1(rdd,count))
ssc.start()
ssc.awaitTermination()
