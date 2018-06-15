#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 20:44:05 2018

@author: chenchacha
"""
import pickle
import numpy as np
import tensorflow as tf
import getdata
print('knn begin!')

X_train = getdata.X_train
X_train_deskew = getdata.X_train_deskew
X_test = getdata.X_test
X_test_deskew = getdata.X_test_deskew
X_train_deskew_reducedn = getdata.X_train_deskew_reducedn
X_test_deskew_reducedn = getdata.X_test_deskew_reducedn
    
y_train = np.fromfile("./dataset/mnist_train_label",dtype=np.uint8)    
y_test = np.fromfile("./dataset/mnist_test_label",dtype=np.uint8)

sess = tf.Session()

np.random.seed(25)  # set seed for reproducibility
train_size = 60000
test_size = 10000
x_vals_train = X_train
x_vals_test = X_test
y_vals_train = y_train
y_vals_test = y_test

k = 5
batch_size=5

with tf.name_scope("IO"):
# Placeholders
    x_data_train = tf.placeholder(shape=[None, 2025], dtype=tf.float32)
    x_data_test = tf.placeholder(shape=[None, 2025], dtype=tf.float32)
    y_target_train = tf.placeholder(shape=[None, 10], dtype=tf.float32)
    y_target_test = tf.placeholder(shape=[None, 10], dtype=tf.float32)

with tf.name_scope("KNN"):    
    #each train and each test dist
    distance = tf.reduce_sum(tf.abs(tf.subtract(x_data_train, tf.expand_dims(x_data_test,1))), axis=2) 
    # Get min distance index (Nearest neighbor)
    top_k_xvals, top_k_indices = tf.nn.top_k(tf.negative(distance), k=k)
    prediction_indices = tf.gather(y_target_train, top_k_indices)
    # Predict the mode category: k nearest nbrs may result in different preds, pick the pred with highest freq
    count_of_predictions = tf.reduce_sum(prediction_indices, axis=1)
    prediction = tf.argmax(count_of_predictions, axis=1)


num_loops = int(np.ceil(len(x_vals_test)/batch_size))
test_output = []
actual_vals = []

with sess.as_default():
    for i in range(num_loops):
        init = tf.global_variables_initializer()
        sess.run(init)
        min_index = i*batch_size
        max_index = min((i+1)*batch_size,len(x_vals_train))
        x_batch = x_vals_test[min_index:max_index] 
        y_batch = y_vals_test[min_index:max_index]
        predictions = sess.run(prediction, feed_dict={x_data_train: x_vals_train, x_data_test: x_batch,
                                             y_target_train: y_vals_train, y_target_test: y_batch})
        test_output.extend(predictions)
        actual_vals.extend(np.argmax(y_batch, axis=1))

    accuracy = sum([1./test_size for i in range(test_size) if test_output[i]==actual_vals[i]])
    print("Accuracy on test data (k=5): %.5f%% " %(accuracy))



x_vals_train = X_train_deskew
x_vals_test = X_test_deskew

with sess.as_default():
    for i in range(num_loops):
        init = tf.global_variables_initializer()
        sess.run(init)
        min_index = i*batch_size
        max_index = min((i+1)*batch_size,len(x_vals_train))
        x_batch = x_vals_test[min_index:max_index] 
        y_batch = y_vals_test[min_index:max_index]
        predictions = sess.run(prediction, feed_dict={x_data_train: x_vals_train, x_data_test: x_batch,
                                             y_target_train: y_vals_train, y_target_test: y_batch})
        test_output.extend(predictions)
        actual_vals.extend(np.argmax(y_batch, axis=1))

    accuracy = sum([1./test_size for i in range(test_size) if test_output[i]==actual_vals[i]])
    print("Accuracy on test data (k=5) (with deskew): %.5f%% " %(accuracy))

x_vals_train = X_train_deskew_reducedn
x_vals_test = X_test_deskew_reducedn

with sess.as_default():
    for i in range(num_loops):
        init = tf.global_variables_initializer()
        sess.run(init)
        min_index = i*batch_size
        max_index = min((i+1)*batch_size,len(x_vals_train))
        x_batch = x_vals_test[min_index:max_index] 
        y_batch = y_vals_test[min_index:max_index]
        predictions = sess.run(prediction, feed_dict={x_data_train: x_vals_train, x_data_test: x_batch,
                                             y_target_train: y_vals_train, y_target_test: y_batch})
        test_output.extend(predictions)
        actual_vals.extend(np.argmax(y_batch, axis=1))

    accuracy = sum([1./test_size for i in range(test_size) if test_output[i]==actual_vals[i]])
    print("Accuracy on test data (k=5) (X_train_deskew_reducedn): %.5f%% " %(accuracy))

