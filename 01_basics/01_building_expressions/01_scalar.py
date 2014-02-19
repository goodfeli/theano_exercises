# Fill in the TODOs in this exercise, then run
# python 01_scalar.py to see if your solution works!
#
import numpy as np
from theano import function
raise NotImplementedError("TODO: add any other imports you need")

def make_scalar():
    """
    Returns a new Theano scalar.
    """

    raise NotImplementedError("TODO: implement this function.")

def log(x):
    """
    Returns the logarithm of a Theano scalar x.
    """

    raise NotImplementedError("TODO: implement this function.")

def add(x, y):
    """
    Adds two theano scalars together and returns the result.
    """

    raise NotImplementedError("TODO: implement this function.")

if __name__ == "__main__":
    a = make_scalar()
    b = make_scalar()
    c = log(b)
    d = add(a, c)
    f = function([a, b], d)
    a = np.cast[a.dtype](1.)
    b = np.cast[b.dtype](2.)
    actual = f(a,b)
    expected = 1. + np.log(2.)
    assert np.allclose(actual, expected)
    print "SUCCESS!"
