"""Additive tree⊕space variance decomposition with a count decoder.

Each gene's latent leaf log-expression is the sum of three independent latent fields,
``z = u + s + e``, observed through a count (or Gaussian) likelihood:

    u ~ Brownian motion on the tree           (variance scale sigma2_phylo)
    s ~ spatial GMRF on the leaf graph         (variance scale sigma2_space)
    e ~ iid leaf residual / nugget             (variance scale sigma2_resid)

We estimate the three variance scales by maximizing the **joint Laplace marginal** of the additive
model (exact when the leaf likelihood is Gaussian; a tight Laplace approximation for log-concave
count decoders) and report the leaf-marginal variance each component contributes, so a gene can be
split into a *heritable* (tree) part and a *niche* (spatial) part. This is the generative
replacement for the descriptive "heritable vs spatially-restricted" scatter, and -- unlike a
tree-only rate -- it does not misattribute spatial structure to fast evolution.

The joint precision is sparse (tree edges + spatial-graph edges + a diagonal, coupled only at the
leaves); v1 factorizes it densely (small trees), and the structure supports a sparse Cholesky for
scale-up. The tree block reuses the BM precision of :class:`scphytr.inference.tree_laplace._TreeModel`.
"""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.sparse.linalg import splu

from .tree_laplace import _TreeModel


def _splu_logdet(Acsc):
    """log|det A| for a sparse SPD matrix via the SuperLU factorization (sum of log|diag(U)|)."""
    lu = splu(sp.csc_matrix(Acsc))
    return float(np.sum(np.log(np.abs(lu.U.diagonal())))), lu


def _mean_diag_inv(Qcsc, n_probe=40, seed=0):
    """Mean diagonal of the inverse of a sparse precision (mean prior marginal variance).

    Exact dense inverse for small graphs; a Hutchinson estimator (Rademacher probes against the
    sparse factorization) for large ones, so the proxy scale stays cheap on big trees.
    """
    n = Qcsc.shape[0]
    if n <= 1200:
        return float(np.mean(np.diag(np.linalg.inv(Qcsc.toarray()))))
    lu = splu(sp.csc_matrix(Qcsc))
    rng = np.random.default_rng(seed)
    acc = 0.0
    for _ in range(n_probe):
        z = rng.integers(0, 2, size=n).astype(float) * 2.0 - 1.0
        acc += float(z @ lu.solve(z))
    return acc / (n_probe * n)


class GaussianLeafObservation:
    """A Gaussian leaf likelihood ``y ~ N(eta, noise)`` -- the directly-observed-trait path and the
    correctness reference (the joint Laplace is then exact)."""

    def __init__(self, y, noise=1.0):
        self.y = np.asarray(y, dtype=float)
        self.noise = float(noise)

    def mode_init(self):
        return self.y.copy()

    def loglik(self, eta):
        r = self.y - eta
        return float(-0.5 * np.sum(r * r) / self.noise
                     - 0.5 * self.y.size * np.log(2 * np.pi * self.noise))

    def grad(self, eta):
        return (self.y - eta) / self.noise

    def neg_hess_diag(self, eta):
        return np.full(eta.shape, 1.0 / self.noise)


class DecompositionResult:
    """Variance components of one gene's tree⊕space decomposition.

    The headline ``v_phylo``/``v_space``/``v_resid`` are the **realized variances of the posterior
    component means** at the leaves (directly comparable to a planted Var(U)/Var(S), and well
    calibrated). ``v_*_scale`` are the prior-marginal-variance proxies (σ²·mean-marginal-var), which
    equal the REML estimates of the same scales -- used for the Gaussian correctness cross-check.
    """

    def __init__(self, posterior, sig2, scale, loglik, converged):
        self.posterior = posterior               # {'u_leaf','s','e'} posterior means at leaves
        vu = float(np.var(posterior["u_leaf"]))
        vs = float(np.var(posterior["s"]))
        ve = float(np.var(posterior["e"]))
        self.v_phylo, self.v_space, self.v_resid = vu, vs, ve
        self.frac_heritable = float(vu / (vu + vs)) if (vu + vs) > 0 else np.nan
        self.v_phylo_scale = float(scale["phylo"])
        self.v_space_scale = float(scale["space"])
        self.frac_heritable_scale = (float(scale["phylo"] / (scale["phylo"] + scale["space"]))
                                     if (scale["phylo"] + scale["space"]) > 0 else np.nan)
        self.sigma2 = sig2                       # raw scales {phylo, space, resid}
        self.loglik = float(loglik)
        self.converged = bool(converged)


def tree_bm_precision(tree):
    """BM precision over the free tree nodes at unit rate, plus leaf bookkeeping.

    Returns ``(Q1 (nf,nf), leaf_free (n_leaves,), nf, mean_leaf_depth, logdet_Q1)`` where ``Q1`` is
    the Gaussian-Markov precision of a unit-rate Brownian motion (root pinned), ``leaf_free`` maps
    each leaf to its row in ``Q1``, and ``mean_leaf_depth`` is the average root-to-leaf path length
    (the mean marginal BM variance per unit rate).
    """
    M = _TreeModel(tree, 0.0, 0.0, 1.0, root_value=0.0)
    free = [i for i in range(M.N) if M.free[i]]
    pos = {i: k for k, i in enumerate(free)}
    nf = len(free)
    rows, cols, vals = [], [], []
    def add(r, c, v):
        rows.append(r); cols.append(c); vals.append(v)
    for i in range(M.N):
        if M.is_root[i]:
            if M.free[i]:
                add(pos[i], pos[i], M.invV[i])
        else:
            pa, iv, ph = M.parent[i], M.invV[i], M.phi[i]
            if M.free[i]:
                add(pos[i], pos[i], iv)
            if M.free[pa]:
                add(pos[pa], pos[pa], ph * ph * iv)
            if M.free[i] and M.free[pa]:
                add(pos[i], pos[pa], -ph * iv)
                add(pos[pa], pos[i], -ph * iv)
    Q1 = sp.coo_matrix((vals, (rows, cols)), shape=(nf, nf)).tocsc()
    leaf_free = np.array([pos[M.leaf_node_idx[L]] for L in range(len(M.leaf_node_idx))], dtype=int)
    depths = []
    for nd in tree.root.get_leaves():
        d, x = 0.0, nd
        while x is not tree.root:
            d += x.dist; x = x.up
        depths.append(d)
    logdet, _ = _splu_logdet(Q1)
    return Q1, leaf_free, nf, float(np.mean(depths)), float(logdet)


def _logdet_chol(H):
    L = np.linalg.cholesky(H)
    return 2.0 * float(np.sum(np.log(np.diag(L))))


def decompose(tree, obs, Q_space1, *, include_residual=True, restarts=1, seed=0,
              max_newton=50, tol=1e-7):
    """Fit the additive tree⊕space (+residual) decomposition for one gene.

    ``obs`` is a leaf observation model (``loglik``/``grad``/``neg_hess_diag``/``mode_init`` on the
    leaf linear predictor ``eta``; e.g. :class:`~scphytr.observation_models.subclonal.SubclonalObservation`
    or :class:`GaussianLeafObservation`). ``Q_space1`` is the unit-scale spatial precision over
    leaves (``pp.spatial_neighbors`` -> ``uns['spatial_graph']['precision']``), in leaf order.
    Returns a :class:`DecompositionResult` (leaf-marginal variance components).

    ``include_residual`` adds an iid leaf nugget. It is weakly identifiable against the spatial
    field (both act only at the leaves), so on clean count data it can absorb genuine niche variance
    and inflate ``frac_heritable``; callers that want the reliable heritable/niche split (e.g.
    ``tl.decompose_variance``) leave it off.
    """
    Q1, leaf_free, nf, mean_depth, logdet_Q1 = tree_bm_precision(tree)
    Qs1 = sp.csc_matrix(Q_space1)
    nL = Qs1.shape[0]
    if leaf_free.shape[0] != nL:
        raise ValueError(f"spatial precision is {nL}x{nL} but the tree has {leaf_free.shape[0]} leaves")
    logdet_Qs1, _ = _splu_logdet(Qs1)
    mean_space_var = _mean_diag_inv(Qs1)                           # mean prior marginal var of s

    nblk = 2 + int(include_residual)
    dim = nf + nL * (nblk - 1)
    # selection A: eta_j = u[leaf_free[j]] + s_j (+ e_j)  -- sparse, one row per leaf
    ridx = np.arange(nL)
    a_rows = [ridx, ridx]
    a_cols = [leaf_free, nf + ridx]
    if include_residual:
        a_rows.append(ridx); a_cols.append(nf + nL + ridx)
    A = sp.csr_matrix((np.ones(nL * nblk), (np.concatenate(a_rows), np.concatenate(a_cols))),
                      shape=(nL, dim))
    AT = A.T.tocsr()

    def prior_precision(s2p, s2s, s2r):
        blocks = [Q1 * (1.0 / s2p), Qs1 * (1.0 / s2s)]
        if include_residual:
            blocks.append(sp.identity(nL, format="csc") * (1.0 / s2r))
        return sp.block_diag(blocks, format="csc")

    def logdet_prior(s2p, s2s, s2r):
        ld = (logdet_Q1 - nf * np.log(s2p)) + (logdet_Qs1 - nL * np.log(s2s))
        if include_residual:
            ld += -nL * np.log(s2r)
        return ld

    leaf_init = obs.mode_init()
    offset = float(np.mean(leaf_init))                # global log-rate; latent is the deviation
    leaf_init0 = leaf_init - offset

    warm = {"x": None}                                # mode of the previous marginal eval

    def fit_mode(Q):
        if warm["x"] is not None:
            x = warm["x"].copy()                      # consecutive evals sit at nearby scales
        else:
            x = np.zeros(dim)
            x[leaf_free] = leaf_init0                 # cold start: u at the (centered) data
        def psi(xx):
            eta = offset + A.dot(xx)
            return -obs.loglik(eta) + 0.5 * float(xx @ (Q.dot(xx)))
        cur = psi(x)
        ok = False
        lu = None
        for _ in range(max_newton):
            eta = offset + A.dot(x)
            W = obs.neg_hess_diag(eta)
            g = obs.grad(eta)
            grad = Q.dot(x) - AT.dot(g)
            H = (Q + AT.dot(sp.diags(W)).dot(A)).tocsc()
            try:
                lu = splu(H)
                step = lu.solve(-grad)
            except RuntimeError:
                break
            t, new = 1.0, cur
            for _ in range(40):
                xn = x + t * step
                new = psi(xn)
                if new <= cur + 1e-12:
                    break
                t *= 0.5
            ok = abs(new - cur) < tol
            x, cur = xn, new
            if ok:
                break
        warm["x"] = x
        return x, lu, ok

    def neg_marginal(logp):
        s2p, s2s = np.exp(logp[0]), np.exp(logp[1])
        s2r = np.exp(logp[2]) if include_residual else 1.0
        Q = prior_precision(s2p, s2s, s2r)
        x, lu, _ = fit_mode(Q)
        if lu is None:
            return 1e18
        eta = offset + A.dot(x)
        ld_H = float(np.sum(np.log(np.abs(lu.U.diagonal()))))    # log|det H| at the mode
        marg = obs.loglik(eta) - 0.5 * float(x @ Q.dot(x)) + 0.5 * logdet_prior(s2p, s2s, s2r) - 0.5 * ld_H
        return -marg if np.isfinite(marg) else 1e18

    rng = np.random.default_rng(seed)
    var0 = max(np.var(leaf_init), 1e-2)
    base = [np.log(var0 / max(mean_depth, 1e-6)), np.log(var0 / max(mean_space_var, 1e-6))]
    if include_residual:
        base.append(np.log(0.1 * var0))
    inits = [np.array(base)] + [np.array(base) + 0.5 * rng.standard_normal(len(base))
                                for _ in range(restarts)]
    best = None
    for p0 in inits:
        res = minimize(neg_marginal, p0, method="Nelder-Mead",
                       options={"xatol": 1e-3, "fatol": 1e-4, "maxiter": 400})
        if best is None or res.fun < best.fun:
            best = res

    s2p, s2s = np.exp(best.x[0]), np.exp(best.x[1])
    s2r = np.exp(best.x[2]) if include_residual else 0.0
    Q = prior_precision(s2p, s2s, s2r if include_residual else 1.0)
    x, _, ok = fit_mode(Q)
    post = {"u_leaf": x[leaf_free], "s": x[nf:nf + nL],
            "e": x[nf + nL:nf + 2 * nL] if include_residual else np.zeros(nL)}
    return DecompositionResult(
        posterior=post,
        sig2={"phylo": float(s2p), "space": float(s2s), "resid": float(s2r)},
        scale={"phylo": s2p * mean_depth, "space": s2s * mean_space_var},
        loglik=-best.fun, converged=ok)


def reml_gaussian_reference(tree, leaf_values, Q_space1, noise=1e-6):
    """Independent closed-form check: leaf marginal logpdf under sigma2p·C_tree + sigma2s·C_space + ...

    Maximizes the exact multivariate-normal marginal of directly-observed leaf values over the
    variance scales (dense, O(n_leaves^3)); used to validate :func:`decompose` on Gaussian data.
    """
    from ..tools.heritability import shared_ancestry_cov
    C_tree, _ = shared_ancestry_cov(tree)
    C_tree = np.asarray(C_tree, dtype=float)
    Qs1 = np.asarray(Q_space1.todense() if hasattr(Q_space1, "todense") else Q_space1, dtype=float)
    C_space = np.linalg.inv(Qs1)
    y = np.asarray(leaf_values, dtype=float)
    y = y - y.mean()
    n = y.size
    mean_depth = float(np.mean(np.diag(C_tree)))
    mean_space_var = float(np.mean(np.diag(C_space)))

    def negll(logp):
        s2p, s2s, s2e = np.exp(logp)
        S = s2p * C_tree + s2s * C_space + (s2e + noise) * np.eye(n)
        sign, ld = np.linalg.slogdet(S)
        return float(0.5 * ld + 0.5 * y @ np.linalg.solve(S, y))

    res = minimize(negll, np.log([np.var(y) / mean_depth, np.var(y) / mean_space_var, 0.1 * np.var(y)]),
                   method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-5, "maxiter": 2000})
    s2p, s2s, s2e = np.exp(res.x)
    return {"v_phylo": s2p * mean_depth, "v_space": s2s * mean_space_var, "v_resid": s2e,
            "frac_heritable": s2p * mean_depth / (s2p * mean_depth + s2s * mean_space_var)}
