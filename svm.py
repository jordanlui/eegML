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
import glob, os
import csv
#from matplotlib.colors import ListedColormap

# ML Parameters
h = .02  # step size in the mesh
#n_neighbors = 15

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]



# Load our data
#x1 = sio.loadmat('../data/seed/processed/de_lds.mat')
#x1 = np.asarray(x1['xmatrix'])
#x2 = sio.loadmat('../data/seed/processed/de_movingave.mat')
#x2 = np.asarray(x2['xmatrix'])
#
## Join our matrices
#x3 = np.concatenate((x1,x2),axis=1)

# Load the big x matrix instead
x3 = sio.loadmat('../data/seed/processed/xbig.mat')
x3 = np.asarray(x3['matrix'])
x = x3[:,0:-1]
y = x3[:,-1]
y = np.reshape(y,(len(y),1))


#y = sio.loadmat('../data/seed/y.mat')
#y=y['y']
#if len(x) < len(y):
#    y=y[0:len(x)] # cut down length of y if our x is shorter

# Manual Cross Validation
number_attempts = 50
performance =[]
percent_training = 0.9 # percent of data we want to train on

# Make a range of slack variables that we will examine 
cvalues = np.power(float(2),(np.arange(-10,-1)))
for cvalue in cvalues:
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
        
        
        lin_clf = svm.LinearSVC(C=cvalue)
        lin_clf.fit(x_train,y_train)
        
        # Test Model
        testdata = lin_clf.predict(x_test)
        testdata = np.reshape(testdata,(len(testdata),1)) # reshape the data
        compare = testdata==y_test
        numcorrect = np.sum(compare)
        accuracy = numcorrect / len(testdata) * 100 # This is the percentage of correct classifications
    #    print 'accuracy encountered is ', accuracy ,'%'
        performance.append(accuracy)
    
    perf_mean = np.mean(performance)
    perf_max = np.max(performance)
    
    print 'Results from running',number_attempts,'trials. Average accuracy', perf_mean,'. Max value', perf_max
    
    filename = 'svmlog.csv'
    file_exists = os.path.isfile(filename)
    with open(filename,'ab') as csvfile:
        fieldnames = ['number trials','mean acc','max acc','features','trials','C','percent train']    
        logwriter = csv.DictWriter(csvfile,fieldnames=fieldnames)
        if not file_exists:
            logwriter.writeheader()    
        logwriter.writerow({'number trials':number_attempts,'mean acc':perf_mean,'max acc':perf_max,'features':x.shape[1],'trials':x.shape[0],'C':cvalue,'percent train':percent_training})