import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import DistanceMetric


#Score from Model used as Metric Function
def Metric(x, y):
	return score

#KNeighbors Classifier algorithm
def KNeighborgs(x,y,m,n):  
	x_train=x
	y_train=y
	x_test=m
	y_test=n
	DistanceMetric.get_metric('pyfunc', func=Metric)
	KNN = KNeighborsClassifier(n_neighbors=3, algorithm='auto', metric=Metric)
	KNN.fit(x_train, y_train)
	predic = KNN.predict(x_test)
	print(accuracy_score(y_test, predic))
	print(y_test)
	print(predic)