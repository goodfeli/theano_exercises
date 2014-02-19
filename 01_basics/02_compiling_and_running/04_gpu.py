# Run this code on the GPU
import numpy as np
from theano import function
import theano.tensor as T

x = T.scalar()
y = T.scalar()
z = x + y

f = function([x, y], z)
if not any('Gpu' in str(node) for node in f.maker.fgraph.apply_nodes):
    raise RuntimeError("No, this is running on the CPU.")
f(np.cast[x.dtype](0.), np.cast[x.dtype](1.))
print "SUCCESS!"
