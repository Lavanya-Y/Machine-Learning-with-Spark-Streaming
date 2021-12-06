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