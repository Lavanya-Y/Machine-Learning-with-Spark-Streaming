import json
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext

spark_context = SparkContext("local[2]", "SparkStreaming")
sql_context = SQLContext(spark_context)
streaming_context = StreamingContext(spark_context, 5)
lines = streaming_context.socketTextStream("localhost", 6100)

def model(df):
	pass
def convertToDf(rdd):
	if not rdd.isEmpty():
		obj = rdd.collect()[0]
		load = json.loads(obj)
		df=sql_context.createDataFrame(load.values(),["Dates","Category","Descript","DayOfWeek","PdDistrict","Resolution","Address","X","Y"])
		model(df)

if lines:
	lines.foreachRDD(lambda x: convertToDf(x))
streaming_context.start()
streaming_context.awaitTermination()
streaming_context.stop()

