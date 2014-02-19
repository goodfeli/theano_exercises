# Figure out why this script doesn't work. Run an experiment to verify
# that your explanation is correct. You probably want to import a
# function that you wrote as part of one of the earlier exercises.

import numpy as np

from theano import function
from theano.printing import Print
import theano.tensor as T

X = T.matrix()
p_tilde = T.exp(X)
p_tilde = Print('p_tilde', attrs=['min', 'max'])(p_tilde)
denom = p_tilde.sum(axis=1, keepdims=True)
p = p_tilde / denom

f = function([X], p)

X = -1000. * np.ones((2, 2)).astype(X.dtype)

output = f(X)

assert np.allclose(output, 0.5 * np.ones((2, 2)))
