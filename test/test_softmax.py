#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 13:40:18 2018

@author: chenchacha
"""
import getdata
import tensorflow as tf
#import pickle
#import numpy as np
from six.moves import xrange
from sklearn.preprocessing import StandardScaler

BATCH_SIZE = 64
NUM_EPOCHS = 100
#EVAL_BATCH_SIZE = 64
#EVAL_FREQUENCY = 100
num_epochs = 10

X_train_deskew_reducedn = getdata.X_train_deskew_reducedn
X_test_deskew_reducedn = getdata.X_test_deskew_reducedn
y_test = getdata.y_test_onehot
y_train = getdata.y_train_onehot
train_size = y_train.shape[0]

scaler = StandardScaler()
scaler.fit(X_train_deskew_reducedn)
X_train_deskew_reducedn_standard=scaler.transform(X_train_deskew_reducedn)

scaler = StandardScaler()
scaler.fit(X_test_deskew_reducedn)
X_test_deskew_reducedn_standard=scaler.transform(X_test_deskew_reducedn)

from sklearn.decomposition import PCA
pca = PCA(n_components = 1215)
X_train_pca = pca.fit_transform(X_train_deskew_reducedn_standard)
X_test_pca = pca.transform(X_test_deskew_reducedn_standard)





###二值化图片淘汰惹

#
#for k in range(60000):
#    mean = X_train_deskew_reducedn_standard[k].mean()
#    for i in range(2025):
#        if ((X_train_deskew_reducedn_standard[k][i])<mean):
#            X_train_deskew_reducedn_standard[k][i]=0
#        else:
#            X_train_deskew_reducedn_standard[k][i]=1
#            
#
#for k in range(10000):
#    mean = X_test_deskew_reducedn_standard[k].mean()
#    for i in range(2025):
#        if ((X_test_deskew_reducedn_standard[k][i])<mean):
#            X_test_deskew_reducedn_standard[k][i]=0
#        else:
#            X_test_deskew_reducedn_standard[k][i]=1


# 模型参数，需要声明为tensorflow变量(tf.Variable)
W = tf.Variable(tf.zeros([1215, 10]))
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
X = tf.placeholder(tf.float32, [None, 1215])
y_ = tf.placeholder(tf.float32, [None, 10])

z = inference(X)
total_loss = loss(X, y_)

# 学习速率，取值过大可能导致算法不能收敛。不同算法可能需要使用的不同值
learning_rate = 0.01

# 使用梯度下降算法寻找损失函数的极小值
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)


correct_prediction = tf.equal(tf.argmax(z, 1), tf.argmax(y_, 1))
evaluate = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


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
        batch_xs = X_train_pca[offset:(offset + BATCH_SIZE), ...]
        batch_ys = y_train[offset:(offset + BATCH_SIZE)]
    # for step in range(training_steps):
    # batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        placeholder_dict = {X: batch_xs, y_: batch_ys}
        sess.run(train_op, feed_dict=placeholder_dict)
#        summary = sess.run(loss_summary, feed_dict=placeholder_dict)
#        writer.add_summary(summary, global_step=step)
    #在测试集上验证模型准确率
    print("Accuracy (deskewd+reduced noise+standard+pca): %.4f" % sess.run(evaluate,feed_dict={X: X_test_pca,y_: y_test}))
#    batch_image = X_test_deskew_reducedn_standard.reshape(10000,45,45)
    #matches = predictions == tf.argmax(y_test)
# later

