# The script does not work because the Print opt disrupts the optimizations.
# This prevents numerical stabilization of the softmax.
# We run two experiments to verify this.
# First, we inspect the compiled graph to verify that it does not use the
# softmax op.
# Second, we run the same functionality without the Print op. We verify that
# the softmax op appears in the compiled graph, and we verify that the new
# graph gets the correct output.
from ex_03_detect_op_soln import contains_softmax

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

assert not contains_softmax(f)

X = -1000. * np.ones((2, 2)).astype(X.dtype)

output = f(X)

assert np.all(np.isnan(output))

X = T.matrix()
p_tilde = T.exp(X)
denom = p_tilde.sum(axis=1, keepdims=True)
p = p_tilde / denom

f = function([X], p)

assert contains_softmax(f)

X = -1000. * np.ones((2, 2)).astype(X.dtype)

output = f(X)

assert np.allclose(output, 0.5 * np.ones((2, 2)))

print "Hypothesis confirmed."
