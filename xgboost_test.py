#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 16:28:35 2018

@author: chenchacha
"""

import xgboost as xgb
import pandas as pd
import numpy as np
#import getdata
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pickle

print(__doc__)

with open('data/data_train.pkl', 'rb') as f:
    X_main_std = pickle.load(f)


    
#with open('data/data_test_reducedn.pkl', 'rb') as f:
#    X_test_deskew_reducedn = pickle.load(f)

#X_main_std = np.fromfile("./data/mnist_train_data",dtype=np.uint8)
y_main = np.fromfile("./data/label_train",dtype=np.uint8)
y_test = np.fromfile("./data/label_test",dtype=np.uint8)
X_main_std = X_main_std.reshape(60000,2025)

print(X_main_std.shape)
print(y_main.shape)

X_train, X_valid, y_train, y_valid = train_test_split(X_main_std, y_main, test_size=0.1)


param_list = [("eta", 0.08), ("max_depth", 6), ("subsample", 0.8), ("colsample_bytree", 0.8), ("objective", "multi:softmax"), ("eval_metric", "merror"), ("alpha", 8), ("lambda", 2), ("num_class", 10)]
n_rounds = 600
early_stopping = 50
    
d_train = xgb.DMatrix(X_train, label=y_train)
d_val = xgb.DMatrix(X_valid, label=y_valid)
eval_list = [(d_train, "train"), (d_val, "validation")]
bst = xgb.train(param_list, d_train, n_rounds, evals=eval_list, early_stopping_rounds=early_stopping, verbose_eval=True)


with open('data/data_test.pkl', 'rb') as f:
    X_test_std = pickle.load(f)
    
d_test = xgb.DMatrix(data=X_test_std)
y_pred = bst.predict(d_test)

np.sum(y_pred == y_test) / y_test.shape