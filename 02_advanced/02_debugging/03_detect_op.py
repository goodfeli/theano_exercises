# Fill in the TODOs and run the script to see if your solution works
from theano import function
from theano import tensor as T

def contains_softmax(f):
    """
    f: a theano function
    Returns True if f contains a T.nnet.Softmax op, False otherwise.
    """

    raise NotImplementedError("TODO: implement this function.")

if __name__ == "__main__":
    X = T.matrix()
    f = function([X], X)
    assert not contains_softmax(f)
    f = function([X], T.nnet.softmax(X))
    assert contains_softmax(f)
    print "SUCCESS!"
