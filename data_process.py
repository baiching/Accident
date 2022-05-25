# -*- coding: utf-8 -*-
"""
Created on Wed May 25 15:34:32 2022

@author: Uthowaipru Chowdhury
"""

import pandas as pd

data_X = pd.read_csv("./data/urbanGB.csv")
data_Y = pd.read_csv("./data/urbanGB.labels.csv")

# data_S = [data_X, data_Y]

# nn = pd.concat(data_S)
# print(nn)