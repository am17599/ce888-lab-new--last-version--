import os, random
import numpy as np
import pandas as pd
import convertToCsv
import TPOT
import KNN
from sklearn.model_selection import train_test_split





def main():

#Image files convert to csv format and store same folder
convertToCsv.create_datasets()
#obtained Traning data and Test data
data_train1=pd.read_csv('training_set.csv')
data_test1=pd.read_csv('test_set.csv')
m_data_train=data_train1.iloc[:,-1:] 
m_data_test=data_test1.iloc[:,-1:] 
data_train=pd.DataFrame(data=data_train1,columns=['path'])
data_train_target=pd.DataFrame(data=m_data_train,columns=['class_id'])
data_test=pd.DataFrame(data=data_test1,columns=['path'])
data_test_target=pd.DataFrame(data=m_data_test,columns=['class_id'])
#Seperate Data
x_train, x_test, y_train, y_test = train_test_split(data_train, data_test,train_size=0.75, test_size=0.25) 


#Process
score=TPOT.Classifier(x_train,y_train):
KNN.KNeighbours(x_train,y_train,x_test,y_test)
TPOT.Classifier.RandomForest(x_train,y_train,x_test,y_test):


if __name__ == "__main__":
    main()

	

