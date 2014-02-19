# Fill in the TODOs and run the script to see if your implementation works.
import numpy as np

from theano import function
from theano import tensor as T

raise NotImplementedError("TODO: add any imports you need.")

class NegativeVariableError(Exception):
    pass

def get_neg_detection_mode():
    """
    Returns a theano Mode that detects if any negative value occurs in the
    evaluation of a theano function.
    This mode should raise a NegativeVariableError if it ever detects any
    variable having a negative value during the execution of the theano
    function.
    """

    raise NotImplementedError("TODO: implement this function.")


if __name__ == "__main__":
    x = T.scalar()
    x.name = 'x'
    y = T.nnet.sigmoid(x)
    y.name = 'y'
    z = - y
    z.name = 'z'
    mode = get_neg_detection_mode()
    f = function([x], z, mode=mode)
    caught = False
    try:
        f(0.)
    except NegativeVariableError:
        caught = True
    if not caught:
        print "You failed to catch a negative value."
        quit(-1)
    f = function([x], y, mode=mode)
    y1 = f(0.)
    f = function([x], y)
    assert np.allclose(f(0.), y1)
    print "SUCCESS!"
