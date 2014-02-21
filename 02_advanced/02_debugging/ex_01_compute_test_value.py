# Run
# python 01_compute_test_value.py
# It should raise an exception when it tries to execute the call to fn.
# The exception doesn't make it easy to tell which line of the python script
# first created an invalid expression though.
# Modify the script to use compute_test_value to find the first bad line.
import numpy as np
from theano import function
from theano import tensor as T
a = T.vector()
b = T.log(a)
c = T.nnet.sigmoid(b)
d = T.sqrt(c)
e = T.concatenate((d, c), axis=0)
f = b * c * d
g = e + f
h = g / c
fn = function([a], h)
fn(np.ones((3,)).astype(a.dtype))
print "SUCCESS!"
