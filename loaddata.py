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


data = sio.loadmat('../data/seed/jianglin_20140404.mat')
labels = sio.loadmat('../data/seed/label.mat')

# Load some data
de_LDS1 = data['de_LDS1']

# Plot
t = np.arange(0,de_LDS1.shape[1])
plt.figure(1)
for channel in range(0,5):
    plt.plot(t,de_LDS1[0,:,channel])