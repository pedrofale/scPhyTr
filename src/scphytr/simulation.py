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


def _bm_field(tree, p, rates, rng, root_value=None):
    """Draw ``p`` independent Brownian motions down the tree (var increment = ``rates``·branch).

    ``rates`` is broadcast to length ``p``. Returns ``(leaf_values (n_leaves, p), leaves)``.
    """
    root = tree.root
    rates = np.broadcast_to(np.asarray(rates, dtype=float).ravel(), (p,))
    Z = {root: (np.zeros(p) if root_value is None else np.asarray(root_value, float).ravel())}
    for nd in root.traverse("preorder"):
        if nd is root:
            continue
        Z[nd] = Z[nd.up] + np.sqrt(np.maximum(rates * nd.dist, 0.0)) * rng.standard_normal(p)
    leaves = root.get_leaves()
    return np.array([Z[l] for l in leaves]), leaves


def simulate_spatial_panel(tree, sigma2_phylo, sigma2_space, dim=2, diffusion=1.0, mu=None,
                           dispersion=None, n_cells=1, mean_size=500.0, n_spatial_basis=8,
                           spatial_lengthscale=0.5, intermixing=0.5, jitter=0.0,
                           spatial_module=None, phylo_module=None, gene_names=None, seed=0):
    """Simulate spatial single-cell lineage data: BM coordinates + additive phylo/niche expression.

    The spatial **coordinates** of each cell are a Brownian motion down the tree (so spatially
    close cells tend to be phylogenetically close -- the real confounding), with per-dimension
    variance ``diffusion``·branch. Each gene's latent log-expression is the **additive** sum of a
    *phylogenetic* component (an independent BM on the tree, variance ``sigma2_phylo``) and a
    *spatial/niche* component (a smooth random field of the coordinates, variance ``sigma2_space``),
    observed as subclonal counts (Poisson, or NB with per-gene ``dispersion``). This is the
    ground-truth generator for the heritable-vs-niche variance decomposition: it stores the planted
    ``var['true_v_phylo','true_v_space','true_frac_heritable']`` and ``obsm['spatial']``.

    ``sigma2_phylo`` / ``sigma2_space`` are per-gene arrays (length ``p``); each component is
    standardized to exactly that variance across leaves, so the targets are clean. Native to
    scPhyTr (node-traversal BM, NB counts) -- no external simulator dependency.
    """
    import anndata as ad
    rng = np.random.default_rng(seed)
    sp_ph = np.asarray(sigma2_phylo, dtype=float).ravel()
    p = sp_ph.size
    sp_sp = np.broadcast_to(np.asarray(sigma2_space, dtype=float).ravel(), (p,))
    mu = np.zeros(p) if mu is None else np.asarray(mu, dtype=float).ravel()

    # (1) spatial coordinates: a mix of Brownian motion down the tree (lineage-determined) and
    #     independent scatter (clonal intermixing). intermixing=0 -> pure lineage (space==tree,
    #     hard / non-identifiable); intermixing=1 -> position independent of lineage (separable).
    coords_bm, leaves = _bm_field(tree, dim, np.full(dim, diffusion), rng)
    coords_bm = (coords_bm - coords_bm.mean(0)) / (coords_bm.std(0) + 1e-9)
    coords_indep = rng.standard_normal((len(leaves), dim))
    rho = float(np.clip(intermixing, 0.0, 1.0))
    coords_leaf = np.sqrt(1.0 - rho) * coords_bm + np.sqrt(rho) * coords_indep
    coords_leaf = (coords_leaf - coords_leaf.mean(0)) / (coords_leaf.std(0) + 1e-9)
    n_leaves = len(leaves)
    names = [l.name for l in leaves]

    # (2) phylogenetic expression component: BM down the tree -> variance sigma2_phylo. Genes
    #     sharing a `phylo_module` id share one BM field (a co-heritable / clonal program).
    if phylo_module is None:
        U, _ = _bm_field(tree, p, np.maximum(sp_ph, 1e-9), rng)
    else:
        pm = np.asarray(phylo_module)
        pfield = {m: _bm_field(tree, 1, 1.0, rng)[0][:, 0] for m in np.unique(pm)}
        U = np.column_stack([pfield[pm[g]] for g in range(p)])
    U = U / (U.std(0) + 1e-9) * np.sqrt(sp_ph)

    # (3) spatial/niche component: a smooth random field over coordinates (random Fourier features)
    def _rff_field():                                          # one smooth random field over coords
        Wf = rng.standard_normal((dim, n_spatial_basis)) / max(spatial_lengthscale, 1e-6)
        bf = rng.uniform(0.0, 2 * np.pi, n_spatial_basis)
        return np.cos(coords_leaf @ Wf + bf) @ rng.standard_normal(n_spatial_basis)

    if spatial_module is None:                                  # independent field per gene
        S = np.column_stack([_rff_field() for _ in range(p)])
    else:                                                       # genes sharing a module id share a field
        sm = np.asarray(spatial_module)
        field = {m: _rff_field() for m in np.unique(sm)}        # one distinct field per module
        S = np.column_stack([field[sm[g]] for g in range(p)])
    S = S / (S.std(0) + 1e-9) * np.sqrt(sp_sp)

    Zleaf = mu + U + S                                          # latent log-expression at leaves

    # (4) observe subclonal counts (cells as replicates of their leaf)
    idx = np.repeat(np.arange(n_leaves), n_cells)
    sizes = rng.gamma(4.0, mean_size / 4.0, size=idx.shape[0]) / mean_size
    lam = (sizes * mean_size)[:, None] * np.exp(Zleaf[idx])
    if dispersion is None:
        Y = rng.poisson(lam)
    else:
        r = np.broadcast_to(np.asarray(dispersion, dtype=float).ravel(), (p,))
        Y = rng.poisson(rng.gamma(r[None, :], lam / r[None, :]))
    cell_coords = coords_leaf[idx]
    if jitter:
        cell_coords = cell_coords + jitter * rng.standard_normal((idx.size, dim))

    A = ad.AnnData(X=Y.astype(float))
    A.var_names = list(gene_names) if gene_names is not None else [f"gene{g}" for g in range(p)]
    A.obs["species"] = [names[i] for i in idx]
    A.obs["size_factors"] = sizes
    A.obsm["spatial"] = cell_coords
    A.var["true_v_phylo"] = sp_ph
    A.var["true_v_space"] = sp_sp
    A.var["true_frac_heritable"] = sp_ph / (sp_ph + sp_sp)
    if spatial_module is not None:
        A.var["true_spatial_module"] = np.asarray(spatial_module)
    if dispersion is not None:
        A.var["true_dispersion"] = np.broadcast_to(np.asarray(dispersion, float).ravel(), (p,))
    A.uns["true_latent"] = Zleaf
    A.uns["true_coords_leaf"] = coords_leaf
    A.uns["true_leaf_names"] = names
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
