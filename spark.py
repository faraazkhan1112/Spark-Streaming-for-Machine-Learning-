from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import Row, SQLContext, SparkSession

conf = SparkConf()
conf.setAppName("BD")
sc = SparkContext(conf=conf)
ssc = StreamingContext(sc, 2)
ssc.checkpoint("checkpoint_BIGDATA")
dataStream = ssc.socketTextStream("localhost",6100)
dataStream.pprint()

ssc.start()
ssc.awaitTermination(100)
ssc.stop()
