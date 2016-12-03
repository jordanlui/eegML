# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 20:57:15 2016

@author: Jordan
"""

# EEG Machine Learning Project

# Initialize
from __future__ import division
import scipy.io as sio
#import matplotlib.pyplot as plt
import numpy as np
#from sklearn import neighbors, datasets
from sklearn import svm
#import glob, os
#import csv
from matplotlib.colors import ListedColormap

# ML Parameters
h = .02  # step size in the mesh
#n_neighbors = 15

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Load our data
x1 = sio.loadmat('../data/seed/processed/de_lds.mat')
x1 = np.asarray(x1['xmatrix'])

x2 = sio.loadmat('../data/seed/processed/de_movingave.mat')
x2 = np.asarray(x2['xmatrix'])


# Join our matrices
x3 = np.concatenate((x1,x2),axis=1)
x = x3

y = sio.loadmat('../data/seed/y.mat')
y=y['y']
if len(x) < len(y):
    y=y[0:len(x)] # cut down length of y if our x is shorter

# Manual Cross Validation
number_attempts = 5
performance =[]
percent_training = 0.9 # percent of data we want to train on

for i in range(0,number_attempts):
    # Shuffle our data
    [x,y] = unison_shuffled_copies(x,y)
    
    # Segment our data to train and test
    
    segment = int(len(x) * percent_training) # Segment of data that we train on, rounded to an integer 
    x_train = x[:segment,:]
    x_test = x[segment:,:]
    y_train = y[:segment,:]
    y_test = y[segment:,:]
    
    # Machine Learning
    # Regular SVM Classification
    #clf = svm.SVC()
    #clf.fit(x_train,y_train)
    
    # Try a Linear Kernel SVM
    
    
    lin_clf = svm.LinearSVC(C=2**-4)
    lin_clf.fit(x_train,y_train)
    
    # Test Model
    testdata = lin_clf.predict(x_test)
    testdata = np.reshape(testdata,(len(testdata),1)) # reshape the data
    compare = testdata==y_test
    numcorrect = np.sum(compare)
    accuracy = numcorrect / len(testdata) * 100 # This is the percentage of correct classifications
#    print 'accuracy encountered is ', accuracy ,'%'
    performance.append(accuracy)

print 'average accuracy was', np.mean(performance)
print 'max value was', np.max(performance)