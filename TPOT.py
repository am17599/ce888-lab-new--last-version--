import numpy as np
from tpot import TPOTClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier



#TPOT Classifier algorithm
def Classifier(x,y):
	x_train=x
	y_train=y
	tpot = TPOTClassifier(verbosity=2, max_time_mins=10, population_size=50,)
	tpot.fit(x_train, y_train)
	tpot.export('tpot_assignment_pipeline.py')
	TPOT_predict = tpot.predict(x_test)
	score = tpot.score(x_test, y_test)
	#print(score)
	#print(y_test)
	#print(TPOT_predict)
	return score
	

#RandomForest model from TPOT pipeline
def RandomForest(x,y,m,n):
	x_train=x
	y_train=y
	x_test=m
	y_test=n
	model = RandomForestClassifier(bootstrap=False, class_weight="balanced" ,max_features=0.7 ,criterion="gini", min_samples_leaf=5, min_samples_split=9,n_estimators=150)
	model.fit(x_train, y_train)
	predic = model.predict(x_test)
	print(accuracy_score(y_test, predic))
	print(y_test)
	print(predic)
