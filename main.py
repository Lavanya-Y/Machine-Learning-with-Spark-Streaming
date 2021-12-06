from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext
from pyspark.ml.linalg import Vector
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer, VectorAssembler
from pyspark.sql.functions import length
import json
import numpy as np
from pyspark.sql import SparkSession
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import scale, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.datasets import make_classification
from pyspark.sql.functions import length
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import MiniBatchKMeans

spark_context = SparkContext("local[2]", "SparkStreaming")
sql_context = SQLContext(spark_context)
streaming_context = StreamingContext(spark_context, 5)
lines = streaming_context.socketTextStream("localhost", 6100)

def preProcess(df):
	df=df.withColumn('len', length(df['Message']))
	spam_indexer = StringIndexer(inputCol='Spam/Ham', outputCol='class')
	token = Tokenizer(inputCol='Message', outputCol='textToken')
	vector_count = CountVectorizer(inputCol='token_stop', outputCol='vector')
	remove_stop = StopWordsRemover(inputCol='textToken', outputCol='token_stop')
	idf = IDF(inputCol='vector', outputCol='tf')
	clean = VectorAssembler(inputCols=['tf', 'len'], outputCol='attribute')
	pipeLine = Pipeline(stages=[spam_indexer, token, remove_stop, vector_count, idf, clean])
	data_new = (pipeLine.fit(df)).transform(df)
	return data_new
	
def naiveBayes(data):
	class_data=np.array(data.select("class").collect())
	array_data =  np.array(data.select("attribute").collect())
	n, n_x, n_y = array_data.shape
	array_data = array_data.reshape((n, n_x*n_y))
	x_train, x_test, y_train, y_test = train_test_split(array_data, class_data, test_size=0.20, random_state=45)
	model = MultinomialNB()
	x=np.unique(y_train)
	model.partial_fit(x_train, y_train, x)
	y_pred=model.predict(x_test)
	accuracy = model.score(x_test,y_test)
	conf = confusion_matrix(y_test, y_pred)
	print("Accuracy Naive Bayes: ", accuracy)
	print(conf)
	#print('Accuracy Naive Bayes:',accuracy_score(y_test, y_pred))

def sgdClassifier(data):
	class_data=np.array(data.select("class").collect())
	array_data =  np.array(data.select("attribute").collect())
	n, n_x, n_y = array_data.shape
	array_data = array_data.reshape((n, n_x*n_y))
	x_train, x_test, y_train, y_test = train_test_split(array_data, class_data, test_size=0.20, random_state=42)
	x=np.unique(y_train)
	sgd = SGDClassifier(max_iter=5, tol=0.01)
	sgd.partial_fit(x_train, y_train, x)
	#y_pred = sgd.predict(x_test)
	accuracy = sgd.score(x_test, y_test)
	print("Accuracy SGD Classifier: ", accuracy)
	
def knn(data):
	class_data=np.array(data.select("class").collect())
	array_data =  np.array(data.select("attribute").collect())
	n, n_x, n_y = array_data.shape
	array_data = array_data.reshape(n, n_x*n_y)
	x_train, x_test, y_train, y_test = train_test_split(array_data, class_data, test_size=0.20, random_state=42)
	model = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
	x=np.unique(y_train)
	model.partial_fit(x_train, y_train, x)
	y_pred = model.predict(x_test)
	accuracy = accuracy_score(x_test,y_test)
	print("Accuracy KNN Classifier: ", accuracy)
	
def kmeans(data):
	class_data=np.array(data.select("class").collect())
	array_data =  np.array(data.select("attribute").collect())
	n, n_x, n_y = array_data.shape
	array_data = array_data.reshape((n, n_x*n_y))
	x_train, x_test, y_train, y_test = train_test_split(array_data, class_data, test_size=0.20, random_state=42)
	model=MiniBatchKMeans(n_clusters=2, random_state=0)
	model=model.partial_fit(x_train)
	print("Accuracy KMeans Clustering:", model.score(x_test, y_test))


def convertToDf(rdd):
	if not rdd.isEmpty():
		obj = rdd.collect()[0]
		load = json.loads(obj)
		df=sql_context.createDataFrame(load.values(),["Subject","Message","Spam/Ham"])
		processed = preProcess(df)
		#naiveBayes(processed)
		knn(processed)
		#sgdClassifier(processed)
		kmeans(processed)
		
if lines:
	lines.foreachRDD(lambda x: convertToDf(x))

streaming_context.start()
streaming_context.awaitTermination()
streaming_context.stop()

