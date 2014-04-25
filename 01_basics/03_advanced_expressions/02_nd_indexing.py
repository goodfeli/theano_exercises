# Fill in the TODOs in this exercise, then run the script to see if your
# solution works.
import numpy as np

from theano import config
from theano import shared
from theano import tensor as T

def shrink_tensor(x, w):
    """
    x : A theano TensorType variable
    w : A theano integer scalar
    Returns:
    y: A theano TensorType variable containing all but the borders
    of x, i.e., discard the first and last w elements along each
    axis of x.

    Examples:
        x = [0, 1, 2, 3, 4], w = 2  -> y = [2]
        x = [[1, 2, 3], [4, 5, 6], [7, 8, 9]], w =1 -> y = [[5]]
    """

    raise NotImplementedError("TODO: implement this function.")


if __name__ == "__main__":
    x = T.TensorType(config.floatX, (False, False, False))()
    xv = np.random.randn(10, 11, 12).astype(config.floatX)
    y = shrink_tensor(x, shared(3)).eval({x : xv})
    assert y.shape == (4, 5, 6), y.shape
    for i in xrange(4):
        for j in xrange(5):
            for k in xrange(6):
                assert y[i, j, k] == xv[i + 3, j + 3, k + 3]
    print "SUCCESS!"
