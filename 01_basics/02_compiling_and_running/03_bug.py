# Something weird happens when you run this code.
# Find something that is not quite right.
# Figure out which compilation modes make the problem more obvious.
# Explain why what is happening is happening.
import numpy as np
from theano import function
from theano import tensor as T
x = T.vector()
y = T.vector()
z = T.zeros_like(y)
a = x + z
f = function([x, y], a)
output = f(np.zeros((1,), dtype=x.dtype), np.zeros((2,), dtype=y.dtype))
