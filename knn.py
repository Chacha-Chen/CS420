#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 20:44:05 2018

@author: chenchacha
"""
import pickle
import numpy as np
import tensorflow as tf


#load data
with open('data/data_train.pkl', 'rb') as f:
    x_train = pickle.load(f)
    
with open('data/data_test.pkl', 'rb') as f:
    x_test = pickle.load(f)
    
y_train = np.fromfile("data/label_train",dtype=np.uint8)    
y_test = np.fromfile("data/label_test",dtype=np.uint8)

x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))

def one_hot(Y,length):
    NewY=[]
    for i in range(len(Y)):
        content=[]
        num = int(Y[i])
        for i in range(num):
            content.append(0)
        content.append(1)
        for i in range(num+1,length):
            content.append(0)
        NewY.append(content)
    return np.array(NewY)

y_test=one_hot(y_test,10)
y_train=one_hot(y_train,10)

# tf Graph Input
#xtr = tf.placeholder("float", [None, 2025])
#xte = tf.placeholder("float", [2025])

#ytr = tf.placeholder(tf.float32, [None, CLASSES])
sess = tf.Session()

np.random.seed(25)  # set seed for reproducibility
train_size = 60000
test_size = 10000
#rand_train_indices = np.random.choice(60000, train_size, replace=False)
#rand_test_indices = np.random.choice(10000, test_size, replace=False)
#x_vals_train = x_train[rand_train_indices]
#x_vals_test = x_test[rand_test_indices]
#y_vals_train = y_train[rand_train_indices]
#y_vals_test = y_test[rand_test_indices]
x_vals_train = x_train
x_vals_test = x_test
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
    
