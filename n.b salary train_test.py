# -*- coding: utf-8 -*-
"""
Created on Thu May 14 15:42:50 2022

@author: lalith kumar
"""
# Preparing a classification model using 'Naive Bayes'.

# importing the data
import pandas as pd

SalaryData_Test = pd.read_csv('C:\\python notes\\ASSIGNMENTS\\naive bayas\\SalaryData_Test.csv')
SalaryData_Train = pd.read_csv('C:\\python notes\\ASSIGNMENTS\\naive bayas\\SalaryData_Train.csv')

SalaryData_Test.head
list(SalaryData_Test)
SalaryData_Test.describe
SalaryData_Test.info()

# string data in dataset.
string_columns=['workclass','education','maritalstatus','occupation','relationship','race','sex','native','Salary']

# finding missing values.
SalaryData_Test.isnull().sum()
SalaryData_Train.isnull().sum()

# applying  label encoding to the object data.

from sklearn import preprocessing
number = preprocessing.LabelEncoder()
for i in string_columns:
    SalaryData_Train[i] = number.fit_transform(SalaryData_Train[i])
    SalaryData_Test[i] = number.fit_transform(SalaryData_Test[i])
    
# split your data in to two part - train and test.
    
colnames = SalaryData_Train.columns
len(colnames[0:13])
trainX = SalaryData_Train[colnames[0:13]]
trainY = SalaryData_Train[colnames[13]]
testX  = SalaryData_Test[colnames[0:13]]
testY  = SalaryData_Test[colnames[13]]

# model developement.
    
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB

sgnb = GaussianNB()
smnb = MultinomialNB()
#  confusion matrix and accuracy.
# gaussian navie bayes.

spred_gnb = sgnb.fit(trainX,trainY).predict(testX)
confusion_matrix(testY,spred_gnb)
# [C.M] array([[10759,   601],
#              [ 2491,  1209]], dtype=int64)

print ("Accuracy",(10759+1209)/(10759+601+2491+1209)) 
# Accuracy 0.7946879150066402

#  confusion matrix and accuracy.
#  multinomial navi bayes.
spred_mnb = smnb.fit(trainX,trainY).predict(testX)
confusion_matrix(testY,spred_mnb)
# [C.M] array([[10891,   469],
#              [ 2920,   780]], dtype=int64)

print("Accuracy",(10891+780)/(10891+780+2920+469))  

#Accuracy 0.7749667994687915
#============================================================================
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
