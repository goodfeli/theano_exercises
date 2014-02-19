from pylearn2.utils import serial
import sys
_, path = sys.argv
model = serial.load(path)
from pylearn2.config import yaml_parse
dataset = yaml_parse.load(model.dataset_yaml_src)
from theano import tensor as T
X = T.matrix()
R = model.fprop(X)
from theano import function
f = function([X], R)
X = dataset.get_batch_design(100)
R = f(X)
X = dataset.get_topological_view(X)
R = dataset.get_topological_view(R)
from pylearn2.gui.patch_viewer import PatchViewer
pv = PatchViewer((10, 20), X.shape[1:3], is_color=False)
for i in xrange(100):
    pv.add_patch(X[i, :, :, :] - 0.5)
    pv.add_patch(R[i, :, :, :] - 0.5)
pv.show()

