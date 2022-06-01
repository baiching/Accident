# -*- coding: utf-8 -*-
"""
Created on Wed May 25 16:05:17 2022

@author: Uthowaipru Chowdhury
"""

#from data_process import CrossValidation
import numpy as np
from data_process import CrossValidation

np.random.seed(0)
W = np.random.randn(2)

#print(len(data_X) - len(data_Y))
X_train, X_test, y_train, y_test = CrossValidation()
m = X_train.shape[0]
print(X_train.shape)

def mean_square_error(W, X, y, m, lambd = 0.01):
    y_hat = np.dot(W.T, X.T)
    l2_reg = lambd * np.sum(W**2)
    return np.sum(np.square(y - y_hat))/m + l2_reg, y_hat

def train(W, X_train, y_train, m, learning_rate=0.01):
    for i in range(50):
        error, y_hat = mean_absolute_error(W, X_train, y_train, m)
        #dW = np.subtract(y_train, y_hat.reshape(len(y_hat), 1) * (- X_train)).mean()*2
        dW = np.absolute(-X_train).mean()
        W = W - learning_rate * dW
        print("epoch "+ str(i)+ " " + str(error))
        
# print(train(W, X_train, y_train, m, 0.01))

def SGD(W, X_train, y_train, m, batch_size, learning_rate=0.01, lambd = 0.01):
    for epoch in range(30):
        error_per_epoch = np.empty((m,2))
        for i in range(m):
            index = np.random.randint(0, m, batch_size)
            X_i = np.take(X_train, indices=index)
            y_i = np.take(y_train, indices=index)
            
            error, y_hat = mean_square_error(W, X_i, y_i, X_i.shape[0], lambd)
            np.append(error_per_epoch, error)
            
            # update
            W = W - learning_rate * (((-2/X_i.shape[0]) * (X_i.dot(y_i - y_hat))) + 2 * lambd * W)
            
        print("epoch " + str(i+1) + " :" + " loss: "+"accuracy :" + str(error_per_epoch.mean()))
        
SGD(W, X_train, y_train, m, 2, 0.01)