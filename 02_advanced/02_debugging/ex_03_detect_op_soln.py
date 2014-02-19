from theano import function
from theano import tensor as T

def contains_softmax(f):
    """
    f: a theano function
    Returns True if f contains a T.nnet.Softmax op, False otherwise.
    """

    apps = f.maker.fgraph.apply_nodes

    for app in apps:
        if isinstance(app.op, T.nnet.Softmax):
            return True
    return False

if __name__ == "__main__":
    X = T.matrix()
    f = function([X], X)
    assert not contains_softmax(f)
    f = function([X], T.nnet.softmax(X))
    assert contains_softmax(f)
    print "SUCCESS!"
