import json
from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import Row, SQLContext, SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import * 
from pyspark.ml.feature import Tokenizer,StopWordsRemover, CountVectorizer,IDF,StringIndexer, HashingTF, NGram, Word2Vec
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.linalg import Vector
from pyspark.ml import Pipeline
from pyspark.ml.pipeline import PipelineModel
from sklearn.naive_bayes import BernoulliNB,MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import r2_score,accuracy_score, precision_score, recall_score, adjusted_rand_score
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import csv
import joblib
schema1 = 'Length long, Subject string, Message string, Spam_Ham string' #SCHEMA FOR THE DATAFRAME DEFINED

def json_data(rdd_data):
  data2 = {}
  for i in rdd_data:
    data2 = json.loads(i, strict = False)
  return data2

def list_to_tuple(list1):
  return tuple(list1)

def log_write(score, acc, pr, re, fscore, path):
	#FUNCTION TO WRITE THE METRICS CALCULATED IN A CSV FILE NAMED AFTER THE MODEL USED
	fields = ['Score', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
	row = [score, acc, pr, re, fscore]
	filename = path[44:48]+".csv"
	with open(filename, 'a') as csvfile:
		csvwriter = csv.writer(csvfile)
		csvwriter.writerow(row)
		
def log_write1(testacc, path):
	#FUNCTION TO WRITE THE PERFORMANCE OF THE ESTIMATOR IN A CSV FILE NAMED AFTER THE MODEL USED
	fields = ['TestAccuracy']
	row = [testacc]
	filename = path[44:48]+".csv"
	with open(filename, 'a') as csvfile:
		csvwriter = csv.writer(csvfile)
		csvwriter.writerow(row)
        
def preprocess(l,sc):
	spark = SparkSession(sc)
	df = spark.createDataFrame(l,schema = schema1)
	
	#PREPROCESSING STARTS
	#MESSAGE COLUMN
	tokenizer = Tokenizer(inputCol="Message", outputCol="token_text")
	stopwords = StopWordsRemover().getStopWords() + ['-']
	stopremove = StopWordsRemover().setStopWords(stopwords).setInputCol('token_text').setOutputCol('stop_tokens')
	bigram = NGram().setN(2).setInputCol('stop_tokens').setOutputCol('bigrams')
	word2Vec = Word2Vec(vectorSize=5, minCount=0, inputCol="bigrams", outputCol="feature2")
	mmscaler = MinMaxScaler(inputCol='feature2',outputCol='scaled_feature2')

	#SUBJECT COLUMN 
	tokenizer1 = Tokenizer(inputCol="Subject", outputCol="token_text1")
	stopwords1 = StopWordsRemover().getStopWords() + ['-']
	stopremove1 = StopWordsRemover().setStopWords(stopwords).setInputCol('token_text1').setOutputCol('stop_tokens1')
	bigram1 = NGram().setN(2).setInputCol('stop_tokens1').setOutputCol('bigrams1')
	word2Vec1 = Word2Vec(vectorSize=5, minCount=0, inputCol="bigrams1", outputCol="feature1")
	mmscaler1 = MinMaxScaler(inputCol='feature1',outputCol='scaled_feature1')

	#ht = HashingTF(inputCol="bigrams", outputCol="ht",numFeatures=8000)
	#CONVERTING THE SPAM/HAM COLUMN TO 0 OR 1
	ham_spam_to_num = StringIndexer(inputCol='Spam_Ham',outputCol='label')
	print("PREPROCESSING STARTS")

	# APPLYING THE PREPROCESSED PIPELINE MODEL ON THE BATCHES RECIEVED
	data_prep_pipe = Pipeline(stages=[ham_spam_to_num,tokenizer,stopremove,bigram,word2Vec,mmscaler,tokenizer1,stopremove1,bigram1,word2Vec1,mmscaler1])
	cleaner = data_prep_pipe.fit(df)
	clean_data = cleaner.transform(df)
	clean_data = clean_data.select(['label','stop_tokens','bigrams','feature1','feature2','scaled_feature2','scaled_feature1'])
	print("PREPROCESSING COMPLETED")

	#SPLITTING INTO TRAINING AND TESTING DATA (0.8,0.2)
	(training,testing) = clean_data.randomSplit([0.8,0.2])
	clean_data.show()
	
	#CONVERTING TRAINING DATA INTO NUMPY ARRAYS
	X_train = np.array(training.select(['scaled_feature1','scaled_feature2']).collect())
	Y_train = np.array(training.select('label').collect())
	print("TEST DATA SPLIT INTO TRAIN AND TEST (0.8,0.2)")
	
	#RESHAPING THE DATA
	nsamples, nx, ny = X_train.shape
	X_train = X_train.reshape((nsamples,nx*ny))
	
	#CONVERTING TESTING DATA INTO NUMPY ARRAYS
	X_test = np.array(testing.select(['scaled_feature1','scaled_feature2']).collect())
	Y_test = np.array(testing.select('label').collect())
	
	#RESHAPING THE DATA
	nsamples, nx, ny = X_test.shape
	X_test = X_test.reshape((nsamples,nx*ny))
	
	return (X_test,Y_test,X_train,Y_train)

def calculatemetrics(Y_test,pred):
    print(pred)
    score = r2_score(Y_test, pred)
    acc = accuracy_score(Y_test, pred)
    pr = precision_score(Y_test, pred)
    re = recall_score(Y_test, pred)
    fscore = (2*re*pr)/(re+pr)
    print("R2 Score: ",score)
    print("Accuracy: ",acc)
    print("Precison: ",pr)
    print("Recall: ",re)
    print("F1 Score: ",fscore)
    return(score, acc, pr, re, fscore)

def calculatetestacc(Y_test,pred):
    print(pred)
    testacc = accuracy_score(Y_test.reshape(-1), pred)
    print("Test Accuracy: ",testacc)
    return(testacc)

#BERNOULLI NAIVE BAYES 
def bnb1(X_test,Y_test,X_train,Y_train,sc):
	#X_test,Y_test,X_train,Y_train = preprocess(l,sc)
	try:
		print("INCREMENTAL LEARNING STARTED (BERNOULLI NAIVE BAYES)")
		classifier1_load = joblib.load('/home/pes1ug19cs153/Desktop/BDProject/build/bNB1.pkl')
		classifier1_load.partial_fit(X_train,Y_train.ravel())
		pred = classifier1_load.predict(X_test)
		score, acc, pr, re, fscore = calculatemetrics(Y_test,pred)
		log_write(score, acc, pr, re, fscore, '/home/pes1ug19cs153/Desktop/BDProject/build/bNB1')
		joblib.dump(classifier1_load, '/home/pes1ug19cs153/Desktop/BDProject/build/bNB1.pkl')
	except Exception as e:
		print("FIRST TRAIN OF MNB MODEL")
		classifier1 = BernoulliNB()
		classifier1.partial_fit(X_train,Y_train.ravel(),classes=np.unique(Y_train))
		pred = classifier1.predict(X_test)
		score, acc, pr, re, fscore = calculatemetrics(Y_test,pred)
		log_write(score, acc, pr, re, fscore, '/home/pes1ug19cs153/Desktop/BDProject/build/bNB1')
		joblib.dump(classifier1, '/home/pes1ug19cs153/Desktop/BDProject/build/bNB1.pkl')
		
#MULTINOMIAL NAIVE BAYES 
def mnb1(X_test,Y_test,X_train,Y_train,sc):
	#X_test,Y_test,X_train,Y_train = preprocess(l,sc)
	try:
		print("INCREMENTAL LEARNING STARTED (MULTINOMIAL NAIVE BAYES)")
		classifier1_load = joblib.load('/home/pes1ug19cs153/Desktop/BDProject/build/mNB1.pkl')
		classifier1_load.partial_fit(X_train,Y_train.ravel())
		pred = classifier1_load.predict(X_test)
		score, acc, pr, re, fscore = calculatemetrics(Y_test,pred)
		log_write(score, acc, pr, re, fscore, '/home/pes1ug19cs153/Desktop/BDProject/build/mNB1')
		joblib.dump(classifier1_load, '/home/pes1ug19cs153/Desktop/BDProject/build/mNB1.pkl')
	except Exception as e:
		print("FIRST TRAIN OF MNB MODEL")
		classifier1 = MultinomialNB()
		classifier1.partial_fit(X_train,Y_train.ravel(),classes=np.unique(Y_train))
		pred = classifier1.predict(X_test)
		score, acc, pr, re, fscore = calculatemetrics(Y_test,pred)
		log_write(score, acc, pr, re, fscore, '/home/pes1ug19cs153/Desktop/BDProject/build/mNB1')
		joblib.dump(classifier1, '/home/pes1ug19cs153/Desktop/BDProject/build/mNB1.pkl')
		
#SGD
def sgd1(X_test,Y_test,X_train,Y_train,sc):
	#X_test,Y_test,X_train,Y_train = preprocess(l,sc)
	try:
		print("INCREMENTAL LEARNING STARTED (SGD)")
		classifier1_load = joblib.load('/home/pes1ug19cs153/Desktop/BDProject/build/sgd1.pkl')
		classifier1_load.partial_fit(X_train,Y_train.ravel())
		pred = classifier1_load.predict(X_test)
		score, acc, pr, re, fscore = calculatemetrics(Y_test,pred)
		log_write(score, acc, pr, re, fscore, '/home/pes1ug19cs153/Desktop/BDProject/build/sgd1')
		joblib.dump(classifier1_load, '/home/pes1ug19cs153/Desktop/BDProject/build/sgd1.pkl')
	except Exception as e:
		print("FIRST TRAIN OF MNB MODEL")
		classifier1 = SGDClassifier()
		classifier1.partial_fit(X_train,Y_train.ravel(),classes=np.unique(Y_train))
		pred = classifier1.predict(X_test)
		score, acc, pr, re, fscore = calculatemetrics(Y_test,pred)
		log_write(score, acc, pr, re, fscore, '/home/pes1ug19cs153/Desktop/BDProject/build/sgd1')
		joblib.dump(classifier1, '/home/pes1ug19cs153/Desktop/BDProject/build/sgd1.pkl')
		
#CLUSTERING
def clustering1(X_test,Y_test,X_train,Y_train,sc):
	#X_test,Y_test,X_train,Y_train = preprocess(l,sc)
	try:
		print("INCREMENTAL LEARNING STARTED (MinBatchKMeans)")
		cls1_load = joblib.load('/home/pes1ug19cs153/Desktop/BDProject/build/cls2.pkl')
		cls1_load.fit(X_train, Y_train.ravel())
		pred = cls1_load.predict(X_test)
		testacc = calculatetestacc(Y_test,pred)
		log_write1(testacc, '/home/pes1ug19cs153/Desktop/BDProject/build/cls2')
		joblib.dump(cls1_load, '/home/pes1ug19cs153/Desktop/BDProject/build/cls2.pkl')
	except Exception as e:
		print("FIRST TRAIN OF CLUSTERING MODEL")
		cls1 = MiniBatchKMeans(n_clusters = 2)
		cls1.fit(X_train,Y_train.ravel())
		pred = cls1.predict(X_test)
		testacc = calculatetestacc(Y_test,pred)
		log_write1(testacc, '/home/pes1ug19cs153/Desktop/BDProject/build/cls2')
		joblib.dump(cls1, '/home/pes1ug19cs153/Desktop/BDProject/build/cls2.pkl')
    
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
		X_test,Y_test,X_train,Y_train = preprocess(rdd2,sc)
		#bnb1(X_test,Y_test,X_train,Y_train,sc)
		mnb1(X_test,Y_test,X_train,Y_train,sc)
		#sgd1(X_test,Y_test,X_train,Y_train,sc)
		#clustering1(X_test,Y_test,X_train,Y_train,sc)
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
