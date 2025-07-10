from ete3 import PhyloTree
import anndata
import scphytr as ph
from scphytr.trait_models.brownian_motion import BrownianMotion
from scphytr.observation_models import poisson
import numpy as np

nwk_txt = "(A:0.1,B:0.2,(C:0.3,D:0.4):0.5);" # Tree
sizes = zip(['A', 'B', 'C', 'D'], [10, 20, 30, 40]) # Number of cells per leaf

# Load the tree
t = PhyloTree(nwk_txt)

# Simulate a trait
bm = BrownianMotion(t, cov_matrix=np.array([[1, 0.5], [0.5, 1]]))
bm.simulate()
ph.pl.trait_value(bm, color='trait')

# Simulate trait values for cells in each leaf
X = poisson.simulate(bm.trait_values, sizes)

# Generate an anndata object
adata = anndata.AnnData(X=X)

# Add the tree to the anndata object
ph.pp.setup_anndata(adata, t)

# Fit the model
ph.tl.estimate_global_rate(adata, 'trait', bm, poisson, method='mcmc', method_kwargs={'n_samples': 1000, 'n_burnin': 100})

# Plot the results
ph.pl.trait_value(adata, color='trait')