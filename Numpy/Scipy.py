# SciPy - FFTpack
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 00:55:23 2020

@author: Vikash Kumar
"""

import numpy as np
list = [1,2,3,4]
arr = np.array(list)
# print(arr)

from numpy import vstack,array
from numpy.random import rand

# data generation with three features
data = vstack((rand(100,3) + array([.5,.5,.5]),rand(100,3)))

from scipy.constants import pi
from math import pi

# print("%.2f"%pi)

#Importing the fft and inverse fft functions from fftpackage
from scipy.fftpack import fft

#create an array with random n numbers
x = np.array([1.0, 2.0, 1.0, -1.0, 1.5])

#Applying the fft function
y = fft(x)
# print (y)

import scipy.integrate
from numpy import exp
f= lambda x:exp(-x**2)
i = scipy.integrate.quad(f, 0, 1)
print(i)

# start from SciPy - Interpolate
# then scikit and matplotlib

# https://www.javatpoint.com/data-structure-tutorial
# https://www.javatpoint.com/ansible-interview-questions




























