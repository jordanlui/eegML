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
n_neighbors = 15

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Load our data
x = np.genfromtxt('../data/seed/x_de_LDS.csv', delimiter=',')
y = sio.loadmat('../data/seed/y.mat')
y=y['y']
y=y[0:660] # cut down length of y if our x is shorter

# Shuffle our data
[x,y] = unison_shuffled_copies(x,y)

# Segment our data
percent_training = 0.9 # percent of data we want to train on
segment = int(len(x) * percent_training) # Segment of data that we train on, rounded to an integer 
x_train = x[:segment,:]
x_test = x[segment:,:]
y_train = y[:segment,:]
y_test = y[segment:,:]

# Machine Learning
# Regular Classification
clf = svm.SVC()
clf.fit(x_train,y_train)


# Test Model
testdata = clf.predict(x_test)
testdata = np.reshape(testdata,(len(testdata),1)) # reshape the data
compare = testdata==y_test
numcorrect = np.sum(compare)
accuracy = numcorrect / len(testdata) * 100 # This is the percentage of correct classifications
print 'accuracy encountered is ', accuracy ,'%'
