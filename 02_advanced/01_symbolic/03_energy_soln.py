import numpy as np

import theano
from theano import function
from theano.sandbox.rng_mrg import MRG_RandomStreams
import theano.tensor as T

def energy(W, V, H):
    """
    W : A theano matrix of RBM weights
        num visible x num hidden
    V : A theano matrix of assignments to visible units
        Each row is another configuration
        Each column corresponds to a different unit
    H : A theano matrix of assignments to hidden units
        Each row is another configuration
        Each column corresponds to a different unit
    Returns:
        E: a theano vector
        Element i gives the energy of configuration (V[i,:], H[i,:])
        (This RBM has no biases, only weights)
    """
    return -(T.dot(V, W) * H).sum(axis=1)

def grad_expected_energy(W, V, H):
    """
    W : A theano matrix of RBM weights
        num visible x num hidden
    V : A theano matrix of samples of visible units
        Each row is another samples
        Each column corresponds to a different unit
    H : A theano matrix of samples of hidden units
        Each row is another samples
        Each column corresponds to a different unit

    Returns:
        dW: a matrix of the derivatives of the expected gradient
            of the energy
    """

    return T.grad(energy(W, V, H).mean(), W, consider_constant=[V, H])


if __name__ == "__main__":
    m = 2
    nv = 3
    nh = 4
    h0 = T.alloc(1., m, nh)
    rng_factory = MRG_RandomStreams(42)
    W = rng_factory.normal(size=(nv, nh), dtype=h0.dtype)
    pv = T.nnet.sigmoid(T.dot(h0, W.T))
    v = rng_factory.binomial(p=pv, size=pv.shape, dtype=W.dtype)
    ph = T.nnet.sigmoid(T.dot(v, W))
    h = rng_factory.binomial(p=ph, size=ph.shape, dtype=W.dtype)

    class _ElemwiseNoGradient(theano.tensor.Elemwise):
        def grad(self, inputs, output_gradients):
            raise TypeError("You shouldn't be differentiating through "
                    "the sampling process.")
            return [ theano.gradient.DisconnectedType()() ]
    block_gradient = _ElemwiseNoGradient(theano.scalar.identity)

    v = block_gradient(v)
    h = block_gradient(h)

    g = grad_expected_energy(W, v, h)
    stats = T.dot(v.T, h) / m
    f = function([], [g, stats])
    g, stats = f()
    assert np.allclose(g, -stats)
    print "SUCCESS!"
