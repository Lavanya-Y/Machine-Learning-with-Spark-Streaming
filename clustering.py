def kmeans(data):
	class_data=np.array(data.select("class").collect())
	array_data =  np.array(data.select("attribute").collect())
	n, n_x, n_y = array_data.shape
	array_data = array_data.reshape((n, n_x*n_y))
	x_train, x_test, y_train, y_test = train_test_split(array_data, class_data, test_size=0.20, random_state=42)
	model=MiniBatchKMeans(n_clusters=2, random_state=0)
	model=model.partial_fit(x_train)
	print("Accuracy KMeans Clustering:", model.score(x_test, y_test))

