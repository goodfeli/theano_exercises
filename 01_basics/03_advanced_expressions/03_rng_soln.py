import numpy as np

from theano import config
from theano import shared

from theano.sandbox.rng_mrg import MRG_RandomStreams


def bernoulli_samples(p):
    """
    p: A theano tensor with elements in the interval [0, 1]
    Returns:
    v: Sampled binary values, with each element of v being 1
       with the probability given by the corresponding element
       of p.
    """

    rng_factory = MRG_RandomStreams(42)
    return rng_factory.binomial(p=p, size=p.shape, dtype=p.dtype, n=1)

if __name__ == "__main__":
    p = shared(np.array(range(11), dtype=config.floatX)/10.)
    s = bernoulli_samples(p).reshape((1, 11))
    m = 100
    samples = np.concatenate([s.eval() for i in xrange(m)], axis=0)
    zeros = samples == 0
    ones = samples == 1
    combined = zeros + ones
    assert combined.min() == 1.
    assert combined.max() == 1.
    mean = samples.mean(axis=0)
    # Unlikely your mean would be off by this much (but possible)
    assert np.abs(mean - p.get_value()).max() < .2
