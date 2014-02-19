# Fill in the TODOs in this exercise, then run
# python 02_vector_mat.py to see if your solution works!
#
import numpy as np
from theano import function
raise NotImplementedError("TODO: add any other imports you need")

def make_vector():
    """
    Returns a new Theano vector.
    """

    raise NotImplementedError("TODO: implement this function.")

def make_matrix():
    """
    Returns a new Theano matrix.
    """

    raise NotImplementedError("TODO: implement this function.")

def elemwise_mul(a, b):
    """
    a: A theano matrix
    b: A theano matrix
    Returns the elementwise product of a and b
    """

    raise NotImplementedError("TODO: implement this function.")

def matrix_vector_mul(a, b):
    """
    a: A theano matrix
    b: A theano vector
    Returns the matrix-vector product of a and b
    """

    raise NotImplementedError("TODO: implement this function.")

if __name__ == "__main__":
    a = make_vector()
    b = make_vector()
    c = elemwise_mul(a, b)
    d = make_matrix()
    e = matrix_vector_mul(d, c)

    f = function([a, b, d], e)

    rng = np.random.RandomState([1, 2, 3])
    a_value = rng.randn(5).astype(a.dtype)
    b_value = rng.rand(5).astype(b.dtype)
    c_value = a_value * b_value
    d_value = rng.randn(5, 5).astype(d.dtype)
    expected = np.dot(d_value, c_value)

    actual = f(a_value, b_value, d_value)

    assert np.allclose(actual, expected)
    print "SUCCESS!"
