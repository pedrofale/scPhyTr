class Base(object):
    def __init__(self, tree, trait_model, observation_model, method_kwargs):
        # An inference approach requires a tree with a trait value for each node
        # The trait model can be a model with per-lineage trait parameters. The learnable parameters are indicated in both the trait model and the observation model.
        # These classes just need to pull them
        self.tree = tree
        self.trait_model = trait_model
        self.observation_model = observation_model
        self.method_kwargs = method_kwargs

        # Get learnable parameters from the trait model and observation model
        self.trait_learnable_parameters = self.trait_model.get_learnable_parameters()
        self.observation_learnable_parameters = self.observation_model.get_learnable_parameters()

    def fit_trait_model(self):
        # Run the inference algorithm to fit the trait model
        pass