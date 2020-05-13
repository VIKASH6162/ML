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
imputer2 = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer2.transform(X[:, 1:3])