# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Data processing
# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('Data.csv')
# print(dataset)
X = dataset.iloc[:,:-1].values 
# here [:,:-1] means take all the columns and values except -1
y = dataset.iloc[:, 3].values

# taking care of missing data
# from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# here axis = 0 for column and 1 for rows
# as we have missing data in columns so we are taking 0 whihc represents columns
# here strategy ="means" it will fill the missing data with mean
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
# print("From missing values", X)

# Encoding categorical data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
# labelencoder_X = LabelEncoder()
# X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# onehotencoder = OneHotEncoder(categorical_features = [0])
# X = onehotencoder.fit_transfor(X).toarrat()
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)
# labelencoder_y = LabelEncoder()
# y = labelencoder_y.fit_transform(y)

# spliting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test =  train_test_split(X,y, test_size = 0.2, 
                                                     random_state = 1)
# feature scalling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# X_train[:, 3:] = sc_X.fit_transform(X_train[:, 3:])
# X_test[:, 3:] = sc_X.transform(X_test[:, 3:])
































