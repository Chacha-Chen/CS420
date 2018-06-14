#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 11:33:59 2018

@author: chenchacha
"""

import numpy as np
import heapq
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import datasets, model_selection
from sklearn.metrics import classification_report
import getdata


y_train = np.fromfile("./data/label_train",dtype=np.uint8)
#target = y_train
y_test = np.fromfile("./data/label_test",dtype=np.uint8)



def cos_knn(k, test_data, test_target, stored_data, stored_target):
    """k: number of neighbors to use for voting
    test_data: a set of unobserved images to classify
    test_target: the labels for the test_data (for calculating accuracy)
    stored_data: the images already observed and available to the model
    stored_target: labels for stored_data
    """

    # find cosine similarity for every point in test_data between every other point in stored_data
    cosim = cosine_similarity(test_data, stored_data)

    # get top k indices of images in stored_data that are most similar to any given test_data point
    top = [(heapq.nlargest((k), range(len(i)), i.take)) for i in cosim]
    # convert indices to numbers using stored target values
    top = [[stored_target[j] for j in i[:k]] for i in top]

    # vote, and return prediction for every image in test_data
    pred = [max(set(i), key=i.count) for i in top]
    pred = np.array(pred)

    # print table giving classifier accuracy using test_target
    print('Accurcy:', classification_report(test_target, pred)) 
    
X_train = getdata.X_train
X_test = getdata.X_test
cos_knn(5, X_test, y_test, X_train, y_train)
del X_train, X_test

X_train_deskew = getdata.X_train_deskew
X_test_deskew = getdata.X_test_deskew
cos_knn(5, X_test_deskew, y_test, X_train_deskew, y_train)
del X_train_deskew, X_test_deskew

X_train_standard=getdata.X_train_standard
X_test_standard=getdata.X_test_standard
cos_knn(5, X_test_standard, y_test, X_train_standard, y_train)
del X_train_standard, X_test_standard

X_train_deskew_reducedn = getdata.X_train_deskew_reducedn
X_test_deskew_reducedn = getdata.X_test_deskew_reducedn
cos_knn(5, X_test_deskew_reducedn, y_test, X_train_deskew_reducedn, y_train)
del X_train_deskew_reducedn, X_test_deskew_reducedn

X_train_deskew_standard=getdata.X_train_deskew_standard
X_test_deskew_standard=getdata.X_test_deskew_standard
cos_knn(5, X_test_deskew_standard, y_test, X_train_deskew_standard, y_train)
del X_train_deskew_standard, X_test_deskew_standard

X_train_deskew_reducedn_standard=getdata.X_train_deskew_reducedn_standard
X_test_deskew_reducedn_standard=getdata.X_test_deskew_reducedn_standard
cos_knn(5, X_test_deskew_reducedn_standard, y_test, X_train_deskew_reducedn_standard, y_train)
del X_train_deskew_reducedn_standard, X_test_deskew_reducedn_standard
