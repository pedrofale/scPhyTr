"""Simulate data from the modular model: a tree × trait model × observation model.

``simulate(tree, trait_model, observation=...)`` draws a latent trait at every node from the
trait model's process (BM / OU / multi-rate BM, read from ``trait_model.process_params()``),
then generates observations at the leaves through the chosen observation model. The observation
is specified by name (matching the ``observation_models`` registry) plus its parameters, and
cells of a leaf are kept as **subclonal replicates** — never pseudobulk.
"""
import numpy as np

from .utils.pruning import _ou_branch


def sample_latent(tree, trait_model, rng):
    """Draw the latent trait value at every node from the trait model's process."""
    p = trait_model.process_params()
    alpha, theta = p["alpha"], p["theta"]
    sigma2, regimes, rates = p.get("sigma2"), p.get("regimes"), p.get("rates")
    root = tree.root
    z = {}
    z[root] = float(p["root_value"]) if p.get("root_value") is not None else \
        float(theta if theta is not None else 0.0)
    for nd in root.traverse("preorder"):
        if nd is root:
            continue
        if rates is not None:                              # multi-rate BM
            phi, v, s2, th = 1.0, nd.dist, float(rates[regimes[nd]]), 0.0
        elif alpha is None or alpha <= 0:                  # BM
            phi, v, s2, th = 1.0, nd.dist, float(sigma2), float(theta)
        else:                                              # OU
            phi, v = _ou_branch(alpha, nd.dist)
            s2, th = float(sigma2), float(theta)
        mean = phi * z[nd.up] + (1.0 - phi) * th
        z[nd] = mean + rng.normal(0.0, np.sqrt(max(v * s2, 0.0)))
    return z


def simulate(tree, trait_model, observation=None, n_cells=1, mean_size=2000.0,
             dispersion=None, obs_sd=1.0, seed=0):
    """Simulate from (tree, trait_model, observation model).

    Parameters
    ----------
    observation : ``None`` (return the latent trait directly), ``"gaussian"`` (add N(0, obs_sd²)),
        or a count model ``"poisson"``/``"subclonal"``/``"negative_binomial"`` with ``n_cells``
        cells per leaf (Poisson, or NB with ``dispersion``).
    Returns a dict with ``leaf_names``, ``latent`` (true per-leaf state), and either ``trait``
    (Gaussian) or the subclonal count fields ``counts``/``leaf_index``/``size_factors``.
    """
    rng = np.random.default_rng(seed)
    z = sample_latent(tree, trait_model, rng)
    leaves = tree.root.get_leaves()
    names = [l.name for l in leaves]
    z_leaf = np.array([z[l] for l in leaves], dtype=float)
    out = {"leaf_names": names, "latent": z_leaf}

    if observation is None:
        out["trait"] = z_leaf
        return out
    if observation == "gaussian":
        out["trait"] = z_leaf + rng.normal(0.0, obs_sd, size=z_leaf.shape)
        return out

    idx = np.repeat(np.arange(len(leaves)), n_cells)
    sizes = rng.gamma(4.0, mean_size / 4.0, size=idx.shape[0]) / mean_size
    lam = (sizes * mean_size) * np.exp(z_leaf[idx])
    if (dispersion is None) and observation in ("poisson", "subclonal"):
        y = rng.poisson(lam)
    else:                                                  # negative-binomial (Gamma-Poisson)
        r = float(dispersion if dispersion is not None else 10.0)
        y = rng.poisson(rng.gamma(r, lam / r))
    out.update(counts=y.astype(float), leaf_index=idx, size_factors=sizes,
               n_leaves=len(leaves))
    return out


def simulate_panel(tree, K, mu=None, dispersion=None, n_cells=4, mean_size=500.0,
                   gene_names=None, seed=0):
    """Simulate a panel of *correlated* genes and return an AnnData ready for ``pp``/``tl``.

    The genes evolve as a multivariate Brownian motion with diffusion matrix ``K`` (its diagonal
    is each gene's rate / heritability, the off-diagonal is gene-gene co-evolution), observed as
    subclonal counts with optional per-gene within-clone negative-binomial dispersion (small
    dispersion ``r`` = plastic). The ground truth is stored for recovery checks:
    ``var['true_rate']`` (diag K), ``var['true_dispersion']``, and ``uns['true_K','true_K_corr']``.
    """
    import anndata as ad
    rng = np.random.default_rng(seed)
    K = np.asarray(K, dtype=float)
    p = K.shape[0]
    mu = np.zeros(p) if mu is None else np.asarray(mu, dtype=float).ravel()
    cholK = np.linalg.cholesky(K + 1e-9 * np.eye(p))
    root = tree.root
    Z = {root: mu.copy()}
    for nd in root.traverse("preorder"):
        if nd is root:
            continue
        Z[nd] = Z[nd.up] + np.sqrt(nd.dist) * (cholK @ rng.standard_normal(p))
    leaves = root.get_leaves()
    names = [l.name for l in leaves]
    Zleaf = np.array([Z[l] for l in leaves])                       # (n_leaves, p)

    idx = np.repeat(np.arange(len(leaves)), n_cells)
    sizes = rng.gamma(4.0, mean_size / 4.0, size=idx.shape[0]) / mean_size
    lam = (sizes * mean_size)[:, None] * np.exp(Zleaf[idx])        # (n_cells_total, p)
    if dispersion is None:
        Y = rng.poisson(lam)
    else:
        r = np.broadcast_to(np.asarray(dispersion, dtype=float).ravel(), (p,))
        Y = rng.poisson(rng.gamma(r[None, :], lam / r[None, :]))

    A = ad.AnnData(X=Y.astype(float))
    A.var_names = list(gene_names) if gene_names is not None else [f"gene{g}" for g in range(p)]
    A.obs["species"] = [names[i] for i in idx]
    A.obs["size_factors"] = sizes
    d = np.sqrt(np.clip(np.diag(K), 1e-12, None))
    A.var["true_rate"] = np.diag(K)
    if dispersion is not None:
        A.var["true_dispersion"] = np.broadcast_to(np.asarray(dispersion, float).ravel(), (p,))
    A.uns["true_K"] = K
    A.uns["true_K_corr"] = K / np.outer(d, d)
    A.uns["true_latent"] = Zleaf
    return A


def simulate_anndata(tree, trait_models, observation="subclonal", n_cells=3, seed=0, **kw):
    """Simulate ``len(trait_models)`` genes and pack them into an AnnData ready for ``pp``.

    ``trait_models`` is a list of fitted/parameterised trait models (one per gene); cells are
    the subclonal replicates (``adata.obs['species']`` = leaf), with size factors set.
    """
    import anndata as ad
    cols, latent = [], []
    leaf_index = size_factors = names = None
    for g, tm in enumerate(trait_models):
        s = simulate(tree, tm, observation=observation, n_cells=n_cells, seed=seed + g, **kw)
        cols.append(s["counts"]); latent.append(s["latent"])
        leaf_index, size_factors, names = s["leaf_index"], s["size_factors"], s["leaf_names"]
    X = np.column_stack(cols)
    A = ad.AnnData(X=X)
    A.var_names = [f"gene{g}" for g in range(len(trait_models))]
    A.obs["species"] = [names[i] for i in leaf_index]
    A.obs["size_factors"] = size_factors
    A.uns["true_latent"] = np.column_stack(latent)
    return A
