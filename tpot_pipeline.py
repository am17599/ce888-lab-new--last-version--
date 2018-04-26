import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split

# Data process 
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=40)

# Results 
exported_pipeline = ExtraTreesClassifier(bootstrap=False,max_features=0.7 ,criterion="gini", min_samples_leaf=5, min_samples_split=9,n_estimators=150)
exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
