# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 20:57:15 2016

@author: Jordan
"""

# EEG Machine Learning Project

# Initialize
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors, datasets
import glob, os
import csv

def DEfunc(x):
    # Accepts a vector and calculates the Differential Entropy
    sigma = np.std(x)
    de = 0.5 * np.log10(2 * 3.14159 * 2.71828 * sigma**2)
    return de
    

#data = [sio.loadmat('../data/seed/jianglin_20140404.mat')]
#data.append(sio.loadmat('../data/seed/jianglin_20140413.mat'))
#data.append(sio.loadmat('../data/seed/jianglin_20140419.mat'))
labels = sio.loadmat('../data/seed/label.mat')

# Directory stuff

os.chdir("../data/seed/")
print len(glob.glob("*[0-9].mat")),' valid files identified in directory'

filelist = glob.glob("*[0-9].mat")

#test = sio.loadmat('../data/seed/liuqiujun_20140621/test.mat')

# Load some data
#de_LDS1 = data[0]['de_LDS1']

# Allocate our variables
x = []
y=[]

# Construct our problem matrix

for trialnum,trialname in enumerate(filelist):
    while True:
        try:
            trial = sio.loadmat(trialname)
            for file in range(0,15): # Loop through the 15 "de_movingAveXX" files for that trial
                filename = 'de_movingAve' + str(file+1)
#                filename = 'de_LDS' + str(file+1)
                de_row = []
        #        Find dimensions of the eeg channels and bands before we loop through them
                channels = trial[filename].shape[0]
                bands = trial[filename].shape[2]
                
                for channel in range(0,channels): # Loop through all of the channels
                    for band in range(0,bands): # Loop through all of the bands
                        de = DEfunc(trial[filename][channel,:,band])
        #                print de
                        de_row.append(de)
                x.append(de_row)
            break
        except ValueError:
            print 'exception encountered with' ,trialname
            break

# Reformat to be array type
x = np.asarray(x)
np.savetxt("x_de_movingAve.csv", x, delimiter=",")

# Create y based on the number of successful x imports
t = (labels['label'])
for i in range(0,15):
    y.append(t[0][i])
while len(y) < 675:
#    print 'y too short. we will append'
    y.append(y)
    # Write to csv file. Having issues with this currently.
#np.savetxt("t3.csv",y,delimiter=",")


# Plot
#t = np.arange(0,de_LDS1.shape[1])
#plt.figure(1)
#for channel in range(0,5):
#    plt.plot(t,de_LDS1[0,:,channel])