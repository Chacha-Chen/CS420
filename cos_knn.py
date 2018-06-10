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

#X_train_deskew = getdata.X_train_deskew
#X_test_deskew = getdata.X_test_deskew
X_train = getdata.X_train
X_train_deskew = getdata.X_train_deskew

X_test = getdata.X_test
X_test_deskew = getdata.X_test_deskew

X_train_deskew_reducedn = getdata.X_train_deskew_reducedn
X_test_deskew_reducedn = getdata.X_test_deskew_reducedn

X_train_standard=getdata.X_train_standard
X_test_standard=getdata.X_test_standard

X_train_deskew_standard=getdata.X_train_deskew_standard
X_test_deskew_standard=getdata.X_test_deskew_standard

X_train_deskew_reducedn_standard=getdata.X_train_deskew_reducedn_standard
X_test_deskew_reducedn_standard=getdata.X_test_deskew_reducedn_standard


# make sure everything was correctly imported
#data.shape, target.shape
# make an array of indices the size of MNIST to use for making the data sets.
# This array is in random order, so we can use it to scramble up the MNIST data
#indx = np.random.choice(len(target), 70000, replace=False)

y_train = np.fromfile("./data/label_train",dtype=np.uint8)
#target = y_train
y_test = np.fromfile("./data/label_test",dtype=np.uint8)

# lets make a dataset of size 50,000, meaning the model will have 50,000 data points to compare each 
# new point it is to classify to
#fifty_x, fifty_y = X_train_deskew[:50000,:],target[:50000]
##fifty_x.shape, fifty_y.shape
#
## lets make one more of size 20,000 and see how classification accuracy decreases when we use that one
#twenty_x, twenty_y = data[50000:,:],target[50000:]
#twenty_x.shape, twenty_y.shape

# build model testing dataset
#X_test_deskew = getdata.X_test_deskew
#test_img1 = np.array(test_img)
#test_img1 = np.array(test_img)
#test_target = [target[i] for i in indx[60000:70000]]
#test_target = y_test
#test_target1 = np.array(test_target)
#test_img.shape, test_target.shape

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
    print('Accurcy:%.4f' % classification_report(test_target, pred)) 
    

cos_knn(5, X_test, y_test, X_train, y_train)
cos_knn(5, X_test_standard, y_test, X_train_standard, y_train)
cos_knn(5, X_test_deskew, y_test, X_train_deskew, y_train)
cos_knn(5, X_test_deskew_standard, y_test, X_train_deskew_standard, y_train)
cos_knn(5, X_test_deskew_reducedn, y_test, X_train_deskew_reducedn, y_train)
cos_knn(5, X_test_deskew_reducedn_standard, y_test, X_train_deskew_reducedn_standard, y_train)
