import numpy as np
from theano import function
import theano.tensor as T

def make_vector():
    """
    Returns a new Theano vector.
    """

    return T.vector()

def make_matrix():
    """
    Returns a new Theano matrix.
    """

    return T.matrix()

def elemwise_mul(a, b):
    """
    a: A theano matrix
    b: A theano matrix
    Returns the elementwise product of a and b
    """

    return a * b

def matrix_vector_mul(a, b):
    """
    a: A theano matrix
    b: A theano vector
    Returns the matrix-vector product of a and b
    """

    return T.dot(a, b)

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
