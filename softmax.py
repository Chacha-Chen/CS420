#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 23:20:33 2018

@author: chenchacha
"""

import tensorflow as tf
#import pickle
#import numpy as np
import getdata
from six.moves import xrange
from sklearn.preprocessing import StandardScaler

#data = [[0, 0], [0, 0], [1, 1], [1, 1]]
#>>> scaler = StandardScaler()
#>>> print(scaler.fit(data))
#StandardScaler(copy=True, with_mean=True, with_std=True)
#>>> print(scaler.mean_)
#[ 0.5  0.5]
#>>> print(scaler.transform(data))

#IMAGE_SIZE = 45
#NUM_CHANNELS = 1
#PIXEL_DEPTH = 255
#NUM_LABELS = 10
#VALIDATION_SIZE = 5000  # Size of the validation set.
#SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 100
#EVAL_BATCH_SIZE = 64
#EVAL_FREQUENCY = 100
num_epochs = 10
X_train = getdata.X_train
X_train_deskew = getdata.X_train_deskew
X_test = getdata.X_test
X_test_deskew = getdata.X_test_deskew
X_train_deskew_reducedn = getdata.X_train_deskew_reducedn
X_test_deskew_reducedn = getdata.X_test_deskew_reducedn
y_test = getdata.y_test_onehot
y_train = getdata.y_train_onehot
train_size = y_train.shape[0]

#print(getdata.X_train.shape)

## 归一化处理
scaler = StandardScaler()
scaler.fit(X_train/255)
X_train_standard=scaler.transform(X_train/255)

scaler = StandardScaler()
scaler.fit(X_test/255)
X_test_standard=scaler.transform(X_test/255)

scaler = StandardScaler()
scaler.fit(X_train_deskew/255)
X_train_deskew_standard=scaler.transform(X_train_deskew/255)

scaler = StandardScaler()
scaler.fit(X_test_deskew/255)
X_test_deskew_standard=scaler.transform(X_test_deskew/255)

scaler = StandardScaler()
scaler.fit(X_train_deskew_reducedn)
X_train_deskew_reducedn_standard=scaler.transform(X_train_deskew_reducedn)

scaler = StandardScaler()
scaler.fit(X_test_deskew_reducedn)
X_test_deskew_reducedn_standard=scaler.transform(X_test_deskew_reducedn)


# 模型参数，需要声明为tensorflow变量(tf.Variable)
W = tf.Variable(tf.zeros([2025, 10]))
b = tf.Variable(tf.zeros([10]))

# 预测函数，根据输入和模型参数计算输出结果。这个函数定义了算法模型，
# 不同算法的区别主要就在这里
def inference(x):
    z = tf.matmul(X, W) + b
    return z

# 损失函数(cost function)，不同算法会使用不同的损失函数，但在这篇
# 文章里都是调用tf提供的库函数，因而区别不大
def loss(x, y):
    z = inference(x)
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=z, labels=y))

# 训练数据，这里使用TensorFlow的占位符机制，其作用类似于函数的形参
X = tf.placeholder(tf.float32, [None, 2025])
y_ = tf.placeholder(tf.float32, [None, 10])

z = inference(X)
total_loss = loss(X, y_)

# 学习速率，取值过大可能导致算法不能收敛。不同算法可能需要使用的不同值
learning_rate = 0.02

# 使用梯度下降算法寻找损失函数的极小值
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

# 验证预测的准确率
#mis_predictions = tf.not_equal(tf.argmax(z, 1), tf.argmax(y_, 1))
correct_prediction = tf.equal(tf.argmax(z, 1), tf.argmax(y_, 1))
evaluate = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 读取数据集，这里tf已经封装好了
#mnist = input_data.read_data_sets("./data", one_hot=True)

# 把loss作为scalar summary写到tf的日志，这样就可以通过tensorboard
# 查看损失函数的变化情况，进行算法调试
writer = tf.summary.FileWriter("./log", graph=tf.get_default_graph())
loss_summary = tf.summary.scalar('Loss', total_loss)

with tf.Session() as sess:
    # 初始化TensorFlow变量，也就是模型参数
    sess.run(tf.global_variables_initializer())

    # 训练模型
    training_steps = 10000
    batch_size = 100
    for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
      # Compute the offset of the current minibatch in the data.
      # Note that we could use better randomization across epochs.
        offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
        batch_xs = X_train[offset:(offset + BATCH_SIZE), ...]
        batch_ys = y_train[offset:(offset + BATCH_SIZE)]
    # for step in range(training_steps):
    # batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        placeholder_dict = {X: batch_xs, y_: batch_ys}
        sess.run(train_op, feed_dict=placeholder_dict)
        summary = sess.run(loss_summary, feed_dict=placeholder_dict)
        writer.add_summary(summary, global_step=step)
    #在测试集上验证模型准确率
    print("Accuracy: %.4f" % sess.run(evaluate,feed_dict={X: X_test,y_: y_test}))

with tf.Session() as sess:
    # 初始化TensorFlow变量，也就是模型参数
    sess.run(tf.global_variables_initializer())

    # 训练模型
    training_steps = 10000
    batch_size = 100
    for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
      # Compute the offset of the current minibatch in the data.
      # Note that we could use better randomization across epochs.
        offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
        batch_xs = X_train_standard[offset:(offset + BATCH_SIZE), ...]
        batch_ys = y_train[offset:(offset + BATCH_SIZE)]
    # for step in range(training_steps):
    # batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        placeholder_dict = {X: batch_xs, y_: batch_ys}
        sess.run(train_op, feed_dict=placeholder_dict)
        summary = sess.run(loss_summary, feed_dict=placeholder_dict)
        writer.add_summary(summary, global_step=step)
    #在测试集上验证模型准确率
    print("Accuracy (standard): %.4f" % sess.run(evaluate,feed_dict={X: X_test_standard,y_: y_test}))
    
  

with tf.Session() as sess:
    # 初始化TensorFlow变量，也就是模型参数
    sess.run(tf.global_variables_initializer())

    # 训练模型
    training_steps = 10000
    batch_size = 100
    for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
      # Compute the offset of the current minibatch in the data.
      # Note that we could use better randomization across epochs.
        offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
        batch_xs = X_train_deskew[offset:(offset + BATCH_SIZE), ...]
        batch_ys = y_train[offset:(offset + BATCH_SIZE)]
    # for step in range(training_steps):
    # batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        placeholder_dict = {X: batch_xs, y_: batch_ys}
        sess.run(train_op, feed_dict=placeholder_dict)
        summary = sess.run(loss_summary, feed_dict=placeholder_dict)
        writer.add_summary(summary, global_step=step)
    #在测试集上验证模型准确率
    print("Accuracy (deskewd): %.4f" % sess.run(evaluate,feed_dict={X: X_test_deskew,y_: y_test}))

with tf.Session() as sess:
    # 初始化TensorFlow变量，也就是模型参数
    sess.run(tf.global_variables_initializer())

    # 训练模型
    training_steps = 10000
    batch_size = 100
    for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
      # Compute the offset of the current minibatch in the data.
      # Note that we could use better randomization across epochs.
        offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
        batch_xs = X_train_deskew_standard[offset:(offset + BATCH_SIZE), ...]
        batch_ys = y_train[offset:(offset + BATCH_SIZE)]
    # for step in range(training_steps):
    # batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        placeholder_dict = {X: batch_xs, y_: batch_ys}
        sess.run(train_op, feed_dict=placeholder_dict)
        summary = sess.run(loss_summary, feed_dict=placeholder_dict)
        writer.add_summary(summary, global_step=step)
    #在测试集上验证模型准确率
    print("Accuracy (deskewd + standard): %.4f" % sess.run(evaluate,feed_dict={X: X_test_deskew_standard,y_: y_test}))



with tf.Session() as sess:
    # 初始化TensorFlow变量，也就是模型参数
    sess.run(tf.global_variables_initializer())

    # 训练模型
    training_steps = 10000
    batch_size = 100
    for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
      # Compute the offset of the current minibatch in the data.
      # Note that we could use better randomization across epochs.
        offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
        batch_xs = X_train_deskew_reducedn[offset:(offset + BATCH_SIZE), ...]
        batch_ys = y_train[offset:(offset + BATCH_SIZE)]
    # for step in range(training_steps):
    # batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        placeholder_dict = {X: batch_xs, y_: batch_ys}
        sess.run(train_op, feed_dict=placeholder_dict)
        summary = sess.run(loss_summary, feed_dict=placeholder_dict)
        writer.add_summary(summary, global_step=step)
    #在测试集上验证模型准确率
    print("Accuracy (deskewd+reduced noise): %.4f" % sess.run(evaluate,feed_dict={X: X_test_deskew_reducedn,y_: y_test}))

with tf.Session() as sess:
    # 初始化TensorFlow变量，也就是模型参数
    sess.run(tf.global_variables_initializer())

    # 训练模型
    training_steps = 10000
    batch_size = 100
    for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
      # Compute the offset of the current minibatch in the data.
      # Note that we could use better randomization across epochs.
        offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
        batch_xs = X_train_deskew_reducedn_standard[offset:(offset + BATCH_SIZE), ...]
        batch_ys = y_train[offset:(offset + BATCH_SIZE)]
    # for step in range(training_steps):
    # batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        placeholder_dict = {X: batch_xs, y_: batch_ys}
        sess.run(train_op, feed_dict=placeholder_dict)
        summary = sess.run(loss_summary, feed_dict=placeholder_dict)
        writer.add_summary(summary, global_step=step)
    #在测试集上验证模型准确率
    print("Accuracy (deskewd+reduced noise+standard): %.4f" % sess.run(evaluate,feed_dict={X: X_test_deskew_reducedn_standard,y_: y_test}))
#    batch_image = X_test_deskew_reducedn_standard.reshape(10000,45,45)
    #matches = predictions == tf.argmax(y_test)
# later
#batch_matches = sess.run(matches, feed_dict={...})
#    from skimage import io
#    batch_matches = sess.run(correct_prediction,feed_dict={X: X_test_deskew_reducedn_standard,y_: y_test})
#    for image, does_match in zip(batch_image, batch_matches):
#      if not does_match:
#        io.imshow(image)


#import numpy as np 
#import matplotlib.pyplot as plt
#index = 0
#misclassifiedIndexes = []
#for label, predict in zip(y_test, predictions):
# if label != predict: 
#  misclassifiedIndexes.append(index)
#  index +=1

# computation graph
    
#predictions = argmax(softmax(final_layer))
#matches = predictions == argmax(labels) # if one-hot encoded
#batch_image = X_test_deskew_reducedn_standard.reshape(10000,45,45)
## later
##batch_matches = sess.run(matches, feed_dict={...})
#from skimage import io
#batch_matches = sess.run(evaluate,feed_dict={X: X_test_deskew_reducedn_standard,y_: y_test})
#for image, does_match in zip(batch_image, batch_matches):
#  if not does_match:
#    io.imshow(image)

#for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
#      # Compute the offset of the current minibatch in the data.
#      # Note that we could use better randomization across epochs.
#      offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
#      batch_data = X_train[offset:(offset + BATCH_SIZE), ...]
#      batch_labels = y_train[offset:(offset + BATCH_SIZE)]
#
#fig, axes = plt.subplots(10, 10, figsize=(10, 10))
#
#for i in range(10):
#    for j in range(10):
#        index=i*10+j
#        axes[i][j].imshow(deskewed_data[index], cmap='gray_r', interpolation='nearest')
#        axes[i][j].set_xticks([])
#        axes[i][j].set_yticks([])