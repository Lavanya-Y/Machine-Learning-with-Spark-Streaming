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
	model.fit(x_train, y_train)
	y_pred = model.predict(x_test)
	accuracy = accuracy_score(x_test,y_test)
	print("Accuracy KNN Classifier: ", accuracy)
