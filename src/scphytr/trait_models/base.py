class BaseTraitModel(object):
    def __init__(self, tree, learnable_parameters=None):
        self.tree = tree
        self.n_species = len(self.tree.phylotree.get_leaf_names())
        self.learnable_parameters = learnable_parameters

    def get_learnable_parameters(self):
        return self.learnable_parameters

    def sample_parameters(self):
        raise NotImplementedError("Subclasses must implement this method")

    def set_parameters(self, params):
        raise NotImplementedError("Subclasses must implement this method")

    def logpdf(self, params):
        raise NotImplementedError("Subclasses must implement this method")

    def score(self):
        raise NotImplementedError("Subclasses must implement this method")