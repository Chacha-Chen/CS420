#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 15:19:17 2018

@author: chenchacha
"""

import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
#from nolearn.dbn import DBN
import timeit
import getdata

import matplotlib.pyplot as plt
import numpy as np
import time
import datetime as dt

from sklearn import datasets, svm, metrics
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


print(__doc__)
def show_some_digits(images, targets, sample_size=24, title_text='Digit {}' ):
    '''
    Visualize random digits in a grid plot
    images - array of flatten gidigs [:,784]
    targets - final labels
    '''
    nsamples=sample_size
    rand_idx = np.random.choice(images.shape[0],nsamples)
    images_and_labels = list(zip(images[rand_idx], targets[rand_idx]))


    img = plt.figure(1, figsize=(15, 12), dpi=160)
    for index, (image, label) in enumerate(images_and_labels):
        plt.subplot(np.ceil(nsamples/6.0), 6, index + 1)
        plt.axis('off')
        #each image is flat, we have to reshape to 2D array 28x28-784
        plt.imshow(image.reshape(28,28), cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title(title_text.format(label))
        
        
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Plots confusion matrix, 
    
    cm - confusion matrix
    """
    plt.figure(1, figsize=(15, 12), dpi=160)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')    
    
    
class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
    
def plot_param_space_scores(scores, C_range, gamma_range):
    """
    Draw heatmap of the validation accuracy as a function of gamma and C
    
    
    Parameters
    ----------
    scores - 2D numpy array with accuracies
    
    """
    #
    # The score are encoded as colors with the hot colormap which varies from dark
    # red to bright yellow. As the most interesting scores are all located in the
    # 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
    # as to make it easier to visualize the small variations of score values in the
    # interesting range while not brutally collapsing all the low score values to
    # the same color.

    
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.jet,
               norm=MidpointNormalize(vmin=0.5, midpoint=0.9))
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Validation accuracy')
    plt.show()
    


y_train = np.fromfile("./data/label_train",dtype=np.uint8)
y_test = np.fromfile("./data/label_test",dtype=np.uint8)

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


####  linear svm with SGD


clf_sgd = SGDClassifier()
clf_sgd.fit(X_train, y_train)
y_pred_sgd = clf_sgd.predict(X_test)
acc_sgd = accuracy_score(y_test, y_pred_sgd)
print("accuracy: ",acc_sgd)

clf_sgd = SGDClassifier()
clf_sgd.fit(X_train_standard, y_train)
y_pred_sgd = clf_sgd.predict(X_test_standard)
acc_sgd = accuracy_score(y_test, y_pred_sgd)
print("accuracy with standard data: ",acc_sgd)

clf_sgd = SGDClassifier()
clf_sgd.fit(X_train_deskew, y_train)
y_pred_sgd = clf_sgd.predict(X_test_deskew)
acc_sgd = accuracy_score(y_test, y_pred_sgd)
print("accuracy with deskewed data: ",acc_sgd)

clf_sgd = SGDClassifier()
clf_sgd.fit(X_train_deskew_standard, y_train)
y_pred_sgd = clf_sgd.predict(X_test_deskew_standard)
acc_sgd = accuracy_score(y_test, y_pred_sgd)
print("accuracy with deskewed + standard data: ",acc_sgd)

clf_sgd = SGDClassifier()
clf_sgd.fit(X_train_deskew_reducedn, y_train)
y_pred_sgd = clf_sgd.predict(X_test_deskew_reducedn)
acc_sgd = accuracy_score(y_test, y_pred_sgd)
print("accuracy with deskewed + reduced noise : ",acc_sgd)

clf_sgd = SGDClassifier()
clf_sgd.fit(X_train_deskew_reducedn_standard, y_train)
y_pred_sgd = clf_sgd.predict(X_test_deskew_reducedn_standard)
acc_sgd = accuracy_score(y_test, y_pred_sgd)
print("accuracy with deskewed + reduced noise +standard: ",acc_sgd)


## linear SVM
clf_svm = LinearSVC()
clf_svm.fit(X_train, y_train)
y_pred_sgd = clf_svm.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_sgd)
print("accuracy: ",acc_svm)

clf_svm = LinearSVC()
clf_svm.fit(X_train_standard, y_train)
y_pred_sgd = clf_svm.predict(X_test_standard)
acc_svm = accuracy_score(y_test, y_pred_sgd)
print("accuracy with standard data: ",acc_svm)

clf_svm = LinearSVC()
clf_svm.fit(X_train_deskew, y_train)
y_pred_sgd = clf_svm.predict(X_test_deskew)
acc_svm = accuracy_score(y_test, y_pred_sgd)
print("accuracy with deskewed data: ",acc_svm)

clf_svm = LinearSVC()
clf_svm.fit(X_train_deskew_standard, y_train)
y_pred_sgd = clf_svm.predict(X_test_deskew_standard)
acc_svm = accuracy_score(y_test, y_pred_sgd)
print("accuracy with deskewed + standard data: ",acc_svm)

clf_svm = LinearSVC()
clf_svm.fit(X_train_deskew_reducedn, y_train)
y_pred_sgd = clf_svm.predict(X_test_deskew_reducedn)
acc_svm = accuracy_score(y_test, y_pred_sgd)
print("accuracy with deskewed + reduced noise : ",acc_svm)

clf_svm = LinearSVC()
clf_svm.fit(X_train_deskew_reducedn_standard, y_train)
y_pred_sgd = clf_svm.predict(X_test_deskew_reducedn_standard)
acc_svm = accuracy_score(y_test, y_pred_sgd)
print("accuracy with deskewed + reduced noise +standard: ",acc_svm)


##SVM with rbf kernel


############### Classification with grid search ##############
# If you don't want to wait, comment this section and uncommnet section below with
# standalone SVM classifier

# Warning! It takes really long time to compute this about 2 days

# Create parameters grid for RBF kernel, we have to set C and gamma
from sklearn.model_selection import GridSearchCV

# generate matrix with all gammas
# [ [10^-4, 2*10^-4, 5*10^-4], 
#   [10^-3, 2*10^-3, 5*10^-3],
#   ......
#   [10^3, 2*10^3, 5*10^3] ]
#gamma_range = np.outer(np.logspace(-4, 3, 8),np.array([1,2, 5]))
gamma_range = np.outer(np.logspace(-3, 0, 4),np.array([1,5]))
gamma_range = gamma_range.flatten()

# generate matrix with all C
#C_range = np.outer(np.logspace(-3, 3, 7),np.array([1,2, 5]))
C_range = np.outer(np.logspace(-1, 1, 3),np.array([1,5]))
# flatten matrix, change to 1D numpy array
C_range = C_range.flatten()

parameters = {'kernel':['rbf'], 'C':C_range, 'gamma': gamma_range}

svm_clsf = svm.SVC()
grid_clsf = GridSearchCV(estimator=svm_clsf,param_grid=parameters,n_jobs=1, verbose=2)


start_time = dt.datetime.now()
print('Start param searching at {}'.format(str(start_time)))

grid_clsf.fit(X_train, y_train)

elapsed_time= dt.datetime.now() - start_time
print('Elapsed time, param searching {}'.format(str(elapsed_time)))
sorted(grid_clsf.cv_results_.keys())

classifier = grid_clsf.best_estimator_
params = grid_clsf.best_params_



scores = grid_clsf.cv_results_['mean_test_score'].reshape(len(C_range),
                                                     len(gamma_range))

plot_param_space_scores(scores, C_range, gamma_range)


######################### end grid section #############


# Now predict the value of the test
expected = y_test
predicted = classifier.predict(X_test)

show_some_digits(X_test,predicted,title_text="Predicted {}")

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
      
cm = metrics.confusion_matrix(expected, predicted)
print("Confusion matrix:\n%s" % cm)

plot_confusion_matrix(cm)

print("Accuracy={}".format(metrics.accuracy_score(expected, predicted)))
