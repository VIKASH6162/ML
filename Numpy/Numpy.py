# 1. NumPy - String Functions
# 2. NumPy - Statistical Functions --> numpy.median()
# 3. 
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 15:24:48 2020

@author: Vikash Kumar
"""


import numpy as np
# 1-D array
a = np.array([1,2,3])

# 2-D array
b = np.array([[1,2,3],['a','b','c']])

# minimum-D
c = np.array([1,2,3,4,5,6,7], ndmin = 3)

# dtype parameter
d = np.array([1,2,3], dtype = int)
# NumPy data types needs to be go through one more time

# ndarray.shape returns a tuple consisting of array dimension
e = np.array([[1,2,3], ['a', 'b', 'c']])

# this resize the ndarray
f = np.array([[1, 2, 3],['a', 'b', 'c']])
f.shape = (3,2)

# reshape function
g = f.reshape(6,1)

# an array of evenly spaced numbers 
h = np.arange(2)

# another exapmle with 1-D to reshaping
i = np.arange(24)
i.ndim
j = i.reshape(2,4,3)

# numpy.itemsize
# This array attribute returns the length of each element of array in bytes.

# dtype of array is int8 (1 byte) 
k = np.array([1, 2, 3, 4, 5], dtype = np.int8)

# dtype of array is now float32 (4 bytes)
l = np.array([1, 2, 3, 4, 5], dtype = np.float32)

# numpy.flags
# shows the current values of flags.

m = np.array([1, 2, 3, 4, 5])

# numpy.empty
# It creates an uninitialized array of specified shape and dtype
n = np.empty([3,2], dtype = int)
# The elements in an array show random values as they are not initialized

# numpy.zeros
# Returns a new array of specified size, filled with zeros.
o = np.zeros([2,3])
p = np.zeros(5, dtype = int)
q = np.zeros(3)
r = np.zeros((2,2), dtype = [('x', 'i4'), ('y', 'i4')])

# numpy.ones
# Returns a new array of specified size and type, filled with ones.
s = np.ones(4)

# numpy.asarray
# useful for converting Python sequence into ndarray
# convert list to ndarray
s = [1, 2, 3]
t = np.asarray(s, dtype = float)

# obtain iterator object from list
list = range(5)
it = iter(list)
# use iterator to create ndarray
u = np.fromiter(it, float)

# numpy.linspace
# This function is similar to arange() function. In this function, instead of 
# step size, the number of evenly spaced values between the interval is specified

v = np.linspace(10, 20, 5)
w = np.linspace(10, 20, 5, endpoint = False)
# 10 11 
# 12 13 
# 14 15 
# 16 17 
# 18 19 
# 20

# NumPy - Indexing & Slicing
x = np.arange(10)
sli = slice(2, 7, 2)

y = x[2:7:2]

# # this returns array of items in the second column
a = np.array([[1,2,3],[3,4,5],[4,5,6]])
# print (a[...,1])

# Now we will slice all items from the second row 
# print (a[1,...])

# Now we will slice all items from column 1 onwards 
# print(a[...,1:])

# Advanced Indexing 
# Integer Indexing
x = np.array([[1, 2], [3, 4], [5, 6]]) 
y = x[[0,1,2], [0,1,0]] 
# print(y)
# (0,0), (1,1) and (2,0)

x = np.array([[ 0,  1,  2],[ 3,  4,  5],[ 6,  7,  8],[ 9, 10, 11]])
rows = np.array([[0,0],[3,3]])
cols = np.array([[0,2],[0,2]]) 
y = x[rows,cols]

# slicing 
z = x[1:4,1:3]

# using advanced index for column 
y = x[1:4,[1,2]]

# Boolean Array Indexing
# This type of advanced indexing is used when the resultant object is meant to 
# be the result of Boolean operations, such as comparison operators.
# Now we will print the items greater than 5 
# print (x[x > 5])

# In this example, NaN (Not a Number) elements are omitted by 
# using ~ (complement operator).
a = np.array([np.nan, 1,2,np.nan,3,4,5]) 
# print( a[~np.isnan(a)])

# how to check array having number or not like nan

# NumPy - Broadcasting
# broadcasting refers to the ability of NumPy to treat arrays of different
# shapes during arithmetic operations.
a = np.array([1,2,3,4]) 
b = np.array([10,20,30,40]) 
c = a * b

a = np.array([[0.0,0.0,0.0],[10.0,10.0,10.0],[20.0,20.0,20.0],[30.0,30.0,30.0]]) 
b = np.array([1.0,2.0,3.0])
# print (a + b)
# print(a*b)

# NumPy - Iterating Over Array
a = np.arange(0,60,5)
a = a.reshape(3,4)
# print(a)
for x in np.nditer(a):
    pass
#    print (x)

# transpose
b = a.T 
for x in np.nditer(b):
    pass
# print(b)

# Modifying Array Values
# print ('Original array is:')
# print (a)

for x in np.nditer(a, op_flags = ['readwrite']):
#   print(id(a[0,0]))
   x[...] = 2*x
#   print(id(x[...]))
#   if x > 60:
#       x[...] = 777
#   print(x)
#   print(a)
# print('Modified array is:')
# print(a) # doubt on how its reflecting in a while I'm changing in x
# x[...] is a copy of every element of array and bcz of that its reflecting there

# External Loop
# In the following example, one-dimensional arrays corresponding to each 
# column is traversed by the iterator

for x in np.nditer(a, flags = ['external_loop'], order = 'F'):
#   print(x,)
    pass

# Broadcasting Iteration
b = np.array([1, 2, 3, 4], dtype = int) 

for x,y in np.nditer([a,b]): 
#   print "%d:%d" % (x,y),
    pass

# numpy.ndarray.flat
a = np.arange(2,10).reshape(4,2) 
# print(a) 

# returns element corresponding to index in flattened array 
# print(a.flat[5])

# numpy.ndarray.flatten
a = np.arange(8).reshape(2,4) 
# print(a.flatten())
# print(a.flatten(order = 'F'))

# numpy.ravel
# print(a.ravel())  
# print(a.ravel(order = 'F'))

# numpy.transpose


# NumPy - bitwise_and
a,b = 13,17 
# print(np.bitwise_and(13, 17))

# similarly bitwise_or

# numpy.invert()
# print(np.invert(np.array([13], dtype = np.uint8)))
# Comparing binary representation of 13 and 242, we find the inversion of bits 

# print('Binary representation of 13:')
# print(np.binary_repr(13, width = 8)) 

# print('Binary representation of 242:') 
# print(np.binary_repr(242, width = 8))

# NumPy - left_shift
# print('Left shift of 10 by two positions:')
# print(np.left_shift(10,2))  

# print('Binary representation of 10:')
# print(np.binary_repr(10, width = 8))  

# print('Binary representation of 40:')
# print(np.binary_repr(40, width = 8))
# Two bits in '00001010' are shifted to left and two 0s appended from right.


# similarly for NumPy - left_shift


# NumPy - Mathematical Functions
a = np.array([0,30,45,60,90])
# print('Sine of different angles:')
# Convert to radians by multiplying with pi/180 
# print(np.sin(a*np.pi/180))
# similarly for cos tan etc

# print('Array containing sine values:')
sin = np.sin(a*np.pi/180) 
# print(sin)

# print('Compute sine inverse of angles. Returned values are in radians.')
inv = np.arcsin(sin) 
# print(inv)

# print('Check result by converting to degrees:')
# print(np.degrees(inv))

# Functions for Rounding
# numpy.around()
a = np.array([1.0,5.55, 123, 0.567, 25.532]) 
# print('Original array:')
# print(a)
# print('After rounding:')
# print(np.around(a))
# print(np.around(a, decimals = 1))
# print(np.around(a, decimals = -1)) # have doubt in this


# numpy.floor()
# returns the largest integer not greater than the input parameter
a = np.array([-1.7, 1.5, -0.2, 0.6, 10]) 
# print(a)
# print(np.floor(a))

# numpy.ceil()
# ceil of the scalar x is the smallest integer i, such that i >= x
a = np.array([-1.7, 1.5, -0.2, 0.6, 10]) 
# print(a)
# print(np.ceil(a))


# NumPy - Arithmetic Operations
# must be either of the same shape or should conform to array broadcasting 
# rules
a = np.arange(9, dtype = np.float_).reshape(3,3)  
# print(a)
b = np.array([10,10,10]) 
# print(b)
# print(np.add(a,b))
# print(np.subtract(a,b))
# print(np.multiply(a,b))
# print(np.divide(a,b))

# numpy.reciprocal()
a = np.array([0.25, 1.33, 1, 0, 100]) 
# print 'Our array is:' 
# print(a)
# print 'After applying reciprocal function:' 
# print(np.reciprocal(a))
b = np.array([100], dtype = int) 
# print 'The second array is:' 
# print(b)
# print 'After applying reciprocal function:' 
# print(np.reciprocal(b))

# numpy.power()
a = np.array([10,100,1000]) 
# print('Our array is:')
# print(a)
# print('Applying power function:')
# print(np.power(a,2))
# print('Second array:')
# b = np.array([1,2,3]) 
# print(b)
# print('Applying power function again:' )
# print(np.power(a,b))

# start from NumPy arithematic Operations numpy.mod()

a = np.array([[3,7,5],[8,4,3],[2,4,9]]) 

# print('Our array is:')
# print(a)

# print('Applying amin() function:')
# print(np.amin(a,1)) # Minima along the second axis which is x

# print('Applying amin() function again:')
# print(np.amin(a,0)) # # Minima along the first axis which is y

# print('Applying amax() function:')
# print(np.amax(a))

# print('Applying amax() function again:')
# print(np.amax(a, axis = 0))

# print('Our array is:' )
# print(a) 
# print('Applying ptp() function:')
# print(np.ptp(a))
# print('Applying ptp() function along axis 1:')
# print(np.ptp(a, axis = 1))
# print('Applying ptp() function along axis 0:')
# print(np.ptp(a, axis = 0)) # range max - min

a = np.array([[30,40,70],[80,20,10],[50,90,60]]) 
# print('Our array is:')
# print(a)
# print('Applying percentile() function:')
# print(np.percentile(a,50))
# print('Applying percentile() function along axis 1:')
# print(np.percentile(a,50, axis = 1))
# print('Applying percentile() function along axis 0:')
# print(np.percentile(a,50, axis = 0))


x = np.array([3, 1, 0]) 
# print(x)

# print('Applying argsort() to x:')
y = np.argsort(x)
# print(y)
# 210 means which is at index 2 will be 1st, whihc is at 1 will be 2nd and 
# which is at 0 will be at third and followed by

# print('Reconstruct original array in sorted order:')
# print(x[y])  
# print('Reconstruct the original array using loop:')
# for i in y: 
#   print(x[i],)


nm = ('raju','anil','ravi','amar') 
dv = ('f.y.', 's.y.', 's.y.', 'f.y.') 
ind = np.lexsort((dv,nm)) 

# print('Applying lexsort() function:')
# print(ind)
from matplotlib import pyplot as plt 

# Compute the x and y coordinates for points on a sine curve 
x = np.arange(0, 3 * np.pi, 0.1) 
y = np.sin(x) 
plt.title("sine wave form") 

# Plot the points using matplotlib 
plt.plot(x, y) 
# plt.show() 

a = np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27]) 
vi = np.histogram(a,bins = [0,20,40,60,80,100]) 
print(vi)
hist,bins = np.histogram(a,bins = [0,20,40,60,80,100]) 
print(hist)
print(bins)















