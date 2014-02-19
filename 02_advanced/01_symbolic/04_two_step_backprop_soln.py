import numpy as np

from theano import config
from theano import function
from theano import shared
from theano import tensor as T

from theano.compat.python2x import OrderedDict

num_vis = 2

class SimpleMLP(object):
    """
    An MLP with one sigmoid hidden layer and one linear output layer
    (for solving regression problems).
    """

    def __init__(self):
        rng = np.random.RandomState([1, 2, 3])
        self.num_hid = 3
        self.W_hid = shared(rng.randn(num_vis, self.num_hid).astype(
            config.floatX))

        self.w_out = shared(rng.randn(self.num_hid).astype(config.floatX))

    def fprop(self, X):
        """
        X : A Theano matrix of input examples.
            Each row is an example.
            Each column is a feature.
        Returns:
            H: A Theano matrix of hidden unit values
            y_hat: A Theano vector of outputs. Output i is the predicted
            value for example i.
        """

        H = T.nnet.sigmoid(T.dot(X, self.W_hid))
        y_hat = T.dot(H, self.w_out)

        return H, y_hat

def loss(y_hat, y):
    """
    y_hat : A minibatch of predictions
    y : A minibatch of targets
    Returns an expression for the loss on this minibatch
    """

    return T.sqr(y_hat - y).mean()

def two_step_backprop(mlp):
    """
    mlp: A SimpleMLP instance
    Returns:
        f1: a theano function
            Takes two arguments: a minibatch of examples and a minibatch of
            targets.
            Returns two values:
                1) The gradient of the loss on mlp.w_out
                2)  An auxiliary value of your choosing
        f2: Takes two arguments: a minibatch of examples, and the auxiliary
            value returned by f1.
            Returns the gradient of the loss on mlp.W_hid
            Should not make use of mlp.w_out at all!
    """

    # Run fprop
    X = T.matrix()
    y = T.vector()
    H, y_hat = mlp.fprop(X)
    l = loss(y_hat, y)

    g_w, g_H = T.grad(l, [mlp.w_out, H])

    f1 = function([X, y], [g_w, g_H])

    known_grads = OrderedDict()
    known_grads[H] = g_H
    g_W = T.grad(None, mlp.W_hid, known_grads=known_grads)

    f2 = function([X, g_H], g_W)

    return f1, f2

if __name__ == "__main__":
    mlp = SimpleMLP()
    X = T.matrix()
    y = T.vector()
    H, y_hat = mlp.fprop(X)
    l = loss(y_hat, y)
    g_W, g_w = T.grad(l, [mlp.W_hid, mlp.w_out])
    rng = np.random.RandomState([1, 2, 3])
    m = 5
    f = function([X, y], [g_W, g_w])
    X = rng.randn(m, num_vis).astype(X.dtype)
    y = rng.randn(m).astype(y.dtype)
    g_W, g_w = f(X, y)
    f1, f2 = two_step_backprop(mlp)
    g_w2, aux = f1(X, y)
    assert np.allclose(g_w, g_w2)
    # Give w_out the wrong size to make sure f2 can't use it
    mlp.w_out.set_value(np.ones(1).astype(mlp.w_out.dtype))
    g_W2 = f2(X, aux)
    assert np.allclose(g_W, g_W2)
    print "SUCCESS!"
