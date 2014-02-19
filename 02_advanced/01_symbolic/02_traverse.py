# Fill in the TODOs and run python 02_traverse.py to see if your solution
# works!
import numpy as np
from theano import tensor as T
raise NotImplementedError("Add any imports you need.")

def arg_to_softmax(prob):
    """
    Oh no! Someone has passed you the probability output,
    "prob", of a softmax function, and you want the unnormalized
    log probability--the argument to the softmax.

    Verify that prob really is the output of a softmax. Raise a
    TypeError if it is not.

    If it is, return the argument to the softmax.
    """

    raise NotImplementedError("Implement this function.")

if __name__ == "__main__":
    x = np.ones((5, 4))
    try:
        arg_to_softmax(x)
        raise Exception("You should have raised an error.")
    except TypeError:
        pass

    x = T.matrix()
    try:
        arg_to_softmax(x)
        raise Exception("You should have raised an error.")
    except TypeError:
        pass

    y = T.nnet.sigmoid(x)
    try:
        arg_to_softmax(y)
        raise Exception("You should have raised an error.")
    except TypeError:
        pass

    y = T.nnet.softmax(x)
    rval = arg_to_softmax(y)
    assert rval is x

    print "SUCCESS!"
