# -*- coding: utf-8 -*-
"""
Created on Wed May 25 15:34:32 2022

@author: Uthowaipru Chowdhury
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data_X = pd.read_csv("./data/urbanGB.csv")
data_Y = pd.read_csv("./data/urbanGB.labels.csv")

#data = pd.read_csv("./data/urbanGBmerged.csv")
mx = MinMaxScaler()
X_minmax = mx.fit_transform(data_X)
Y_minmax = mx.fit_transform(data_Y)

#print(X_train_minmax)
def CrossValidation():
    """
    Returns (X_train, X_test, y_train, y_test)
    -------
    This functions calculates the cross validation for the dataset
    Spliting data by 90/10 for training and test data 

    """
    return train_test_split(X_minmax, Y_minmax, test_size=0.1, random_state=0)

