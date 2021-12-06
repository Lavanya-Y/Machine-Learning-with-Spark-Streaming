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
from classifiers import naiveBayes, knn, sgdClassifier
from preprocessing import preProcess
from clustering import kmeans

spark_context = SparkContext("local[2]", "SparkStreaming")
sql_context = SQLContext(spark_context)
streaming_context = StreamingContext(spark_context, 5)
lines = streaming_context.socketTextStream("localhost", 6100)

def convertToDf(rdd):
	if not rdd.isEmpty():
		obj = rdd.collect()[0]
		load = json.loads(obj)
		df=sql_context.createDataFrame(load.values(),["Subject","Message","Spam/Ham"])
		processed = preProcess(df)
		naiveBayes(processed)
		knn(processed)
		sgdClassifier(processed)
		kmeans(processed)
		
if lines:
	lines.foreachRDD(lambda x: convertToDf(x))

streaming_context.start()
streaming_context.awaitTermination()
streaming_context.stop()

