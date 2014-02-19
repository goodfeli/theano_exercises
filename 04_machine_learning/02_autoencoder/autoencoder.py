from pylearn2.models.mlp import MLP

class Autoencoder(MLP):
    """
    An MLP whose output domain is the same as its input domain.
    """

    def get_target_source(self):
        return 'features'

