#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 16:42:09 2018

@author: chenchacha
"""

import tensorflow as tf
import numpy as np
import getdata
#import tensorflow as tf
import functools

def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper

class MultiSVC(object):
    
    def __init__(self,
                learning_rate = 0.001,
                training_epoch = None,
                error = 0.001,
                display_step = 5):
        self.learning_rate = learning_rate
        self.training_epoch = training_epoch
        self.display_step = display_step
        self.error = error
        
    def __Preprocessing(self, trainX, trainY):
        self.row = row = trainX.shape[0]
        col = trainX.shape[1]
        self.ycol = ycol = trainY.shape[1]
        self.X = tf.placeholder(shape=[row, col], dtype= tf.float32)
        self.Y = tf.placeholder(shape=[row, ycol], dtype= tf.float32)
        self.test = tf.placeholder(shape=[None, col], dtype= tf.float32)
        self.beta = tf.Variable(tf.truncated_normal(shape=[ycol, row], stddev=.1))
    
    def __dense_to_one_hot(self,labels_dense):
        """标签 转换one hot 编码
        输入labels_dense 必须为非负数
        """
        num_classes = len(np.unique(labels_dense)) # np.unique 去掉重复函数
        raws_labels = labels_dense.shape[0]
        index_offset = np.arange(raws_labels) * num_classes
        labels_one_hot = np.zeros((raws_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot      
    
    @lazy_property 
    def Kernel_Train(self):
        tmp_abs = tf.reshape(tensor=tf.reduce_sum(tf.square(self.X), axis=1), shape=[-1,1])
        tmp_ = tf.add(tf.sub(tmp_abs, tf.mul(2., tf.matmul(self.X, tf.transpose(self.X)))), tf.transpose(tmp_abs))
        return tf.exp(tf.mul(self.gamma, tf.abs(tmp_)))
    
    @lazy_property  
    def Cost(self):
        left = tf.reduce_sum(self.beta)
        beta_square = tf.matmul(self.beta, self.beta, transpose_a=True)
        tmp = tf.expand_dims(self.Y, axis= 0)
        Y_square = tf.matmul(tf.transpose(tmp, perm=[2,1,0]), tf.transpose(tmp, perm=[2,0,1]))
        right = tf.reduce_sum(tf.mul(self.Kernel_Train, tf.mul(beta_square, Y_square)), axis=[1,2])
        return tf.reduce_sum(tf.neg(tf.sub(left, right)))
    
    @lazy_property  
    def Kernel_Prediction(self):        
        tmpA = tf.reshape(tf.reduce_sum(tf.square(self.X), 1),[-1,1])
        tmpB = tf.reshape(tf.reduce_sum(tf.square(self.test), 1),[-1,1])
        tmp = tf.add(tf.sub(tmpA, tf.mul(2.,tf.matmul(self.X, self.test, transpose_b=True))), tf.transpose(tmpB))
        return tf.exp(tf.mul(self.gamma, tf.abs(tmp)))
    
    @lazy_property 
    def Prediction(self):
        kernel_out = tf.matmul(tf.mul(tf.transpose(self.Y),self.beta), self.Kernel_Prediction)
        return tf.arg_max(kernel_out - tf.expand_dims(tf.reduce_mean(kernel_out,1),1),0)
    
    @lazy_property 
    def Accuracy(self):
        return tf.reduce_mean(tf.cast(tf.equal(self.Prediction, tf.argmax(self.Y,1)), tf.float32))
    
    def fit(self, trainX, trainY, gamma= 50.):
        trainY = self.__dense_to_one_hot(trainY)
        trainY = np.where(trainY, 1, -1)
        self.sess = tf.InteractiveSession()
        self.__Preprocessing(trainX, trainY)
        self.gamma = tf.constant(value= -gamma, dtype=tf.float32)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.Cost)
        self.sess.run(tf.global_variables_initializer())
        
        if self.training_epoch is not None:        
            for ep in range(self.training_epoch):
                self.sess.run(self.optimizer, feed_dict={self.X:trainX, self.Y:trainY})
                if ep % self.display_step== 0:
                    loss, acc = self.sess.run([self.Cost, self.Accuracy], feed_dict={self.X:trainX, self.Y:trainY, self.test:trainX})
                    print ('epoch=',ep,'loss= ',loss, 'accuracy= ', acc)
        elif self.training_epoch is None:
            acc = 0.1
            ep = 0
            while (acc< 1.- self.error):
                acc,_ = self.sess.run([self.Accuracy, self.optimizer], feed_dict={self.X:trainX, self.Y:trainY, self.test:trainX})
                ep += 1
                if ep % self.display_step== 0: 
                    loss = self.sess.run(self.Cost, feed_dict={self.X:trainX, self.Y:trainY})
                    print ('epoch=',ep,'loss= ',loss, 'accuracy= ', acc)    
        print("Optimization Finished!")      
        self.trainX = trainX
        self.trainY = trainY
    
    def pred(self,test):
        output = self.sess.run(self.Prediction, feed_dict={self.X:self.trainX, self.Y:self.trainY, self.test:test})
        return output
       
    
X_train = getdata.X_train
X_train_deskew = getdata.X_train_deskew
X_test = getdata.X_test
X_test_deskew = getdata.X_test_deskew
X_train_deskew_reducedn = getdata.X_train_deskew_reducedn
X_test_deskew_reducedn = getdata.X_test_deskew_reducedn
y_test = getdata.y_test_onehot
y_train = getdata.y_train_onehot

data = X_train_deskew_reducedn
target = y_train
test = X_test_deskew_reducedn

if __name__ == '__main__':
    data = tf.placeholder(tf.float32, [None, 45, 45])
    target = tf.placeholder(tf.float32, [None, 1])
    svm_model = MultiSVC(training_epoch=5)
    svm_model.fit(data,target)
    print(svm_model.pred(test))
    
