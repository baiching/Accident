# -*- coding: utf-8 -*-
"""
Created on Wed May 25 16:05:17 2022

@author: Uthowaipru Chowdhury
"""

from data_process import data_X, data_Y
import numpy as np

np.random.seed(0)
W = np.random.randn(2)

#print(len(data_X) - len(data_Y))


def Ein(W, X, y):
    y_hat = np.dot(W.T, X.T)
    return np.subtract(data_Y, y_hat.reshape(len(data_X), 1))

print(Ein(W, data_X, data_Y))