# Fill in the TODOs in this exercise, then run
# python 03_tensor.py to see if your solution works!
import numpy as np
from theano import function
raise NotImplementedError("TODO: add any other imports you need")

def make_tensor(dim):
    """
    Returns a new Theano tensor with no broadcastable dimensions.
    dim: the total number of dimensions of the tensor.
    (You can use any dtype you like)
    """

    raise NotImplementedError("TODO: implement this function.")

def broadcasted_add(a, b):
    """
    a: a 3D theano tensor
    b: a 4D theano tensor
    Returns c, a 4D theano tensor, where

    c[i, j, k, l] = a[l, k, i] + b[i, j, k, l]

    for all i, j, k, l
    """

    raise NotImplementedError("TODO: implement this function.")

def partial_max(a):
    """
    a: a 4D theano tensor

    Returns b, a theano matrix, where

    b[i, j] = max_{k,l} a[i, k, l, j]

    for all i, j
    """

    raise NotImplementedError("TODO: implement this function.")

if __name__ == "__main__":
    a = make_tensor(3)
    b = make_tensor(4)
    c = broadcasted_add(a, b)
    d = partial_max(c)

    f = function([a, b,], d)

    rng = np.random.RandomState([1, 2, 3])
    a_value = rng.randn(2, 2, 2).astype(a.dtype)
    b_value = rng.rand(2, 2, 2, 2).astype(b.dtype)
    c_value = np.transpose(a_value, (2, 1, 0))[:, None, :, :] + b_value
    expected = c_value.max(axis=1).max(axis=1)

    actual = f(a_value, b_value)

    assert np.allclose(actual, expected), (actual, expected)
    print "SUCCESS!"
