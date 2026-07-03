"""O(n) Laplace-approximate marginal likelihood for latent tree models.

The trait is a latent value at *every* node of the tree, evolving along edges by a
BM/OU linear-Gaussian transition, and observed (through any per-leaf likelihood)
only at the leaves. The joint prior is a Gaussian Markov random field whose
precision Q is tree-structured (each node couples only to its parent). For a
non-conjugate leaf likelihood the Laplace marginal needs the posterior mode and
the log-determinant of (Q + W), where W is the diagonal observation curvature.
Both are obtained by Gaussian elimination in post-order over the tree -- the
linear-Gaussian generalization of Felsenstein's pruning -- in O(n) time, never
forming a dense covariance.

Because Laplace is exact along the (Gaussian) internal-node directions, this
returns exactly the same value as marginalizing the internal nodes analytically
and applying Laplace only at the leaves (the dense O(n^3) computation), but in
linear time.

The observation model is any object exposing, on the leaf-latent vector f:
    loglik(f) -> float, grad(f) -> (n,), neg_hess_diag(f) -> (n,)>=0, mode_init() -> (n,)
with leaves ordered as ``tree.phylotree.get_leaf_names()``.
"""

import numpy as np


def _ou_branch(alpha, t):
    """Contraction phi = e^{-alpha t} and variance factor v(t); BM when alpha<=0."""
    if alpha is None or alpha <= 0:
        return 1.0, t
    phi = np.exp(-alpha * t)
    v = -np.expm1(-2.0 * alpha * t) / (2.0 * alpha)
    return phi, v


class _TreeModel:
    """Precomputed per-node BM/OU edge quantities for the latent tree Gaussian.

    A node whose edge variance is ~0 (a zero-length root branch => fixed ancestral
    state) is treated as *fixed* rather than a free latent: it is pinned to its
    prior mean and excluded from the elimination and the determinants, while its
    children correctly see it as a known constant.
    """

    _ZERO = 1e-12

    def __init__(self, tree, alpha, theta, sigma2, regimes=None, root_value=None):
        # --- structure (topology + branch lengths); independent of the rate params, built once ---
        root = tree.root
        self.post = list(root.traverse("postorder"))   # children before parents
        self.pre = list(root.traverse("preorder"))
        self.index = {nd: i for i, nd in enumerate(self.post)}
        N = len(self.post)
        self.N = N
        self.parent = np.full(N, -1, dtype=int)         # post-order index of the parent (-1 at root)
        self.dist = np.zeros(N)                         # branch length above each node
        self.is_root = np.zeros(N, dtype=bool)
        for nd in self.post:
            i = self.index[nd]
            self.dist[i] = nd.dist
            if nd is root:
                self.is_root[i] = True
            else:
                self.parent[i] = self.index[nd.up]
        self._nonroot = np.where(~self.is_root)[0]
        self._nr_parent = self.parent[self._nonroot]
        # regime index per node (for per-regime theta); None => single optimum
        if regimes is None:
            self._regime_idx = None
        else:
            self._regime_idx = np.array([regimes[nd] for nd in self.post], dtype=int)
        # observation/leaf order -> node index
        node_by_name = {nd.name: self.index[nd] for nd in self.post if nd.is_leaf()}
        self.leaf_node_idx = np.array([node_by_name[name] for name in tree.phylotree.get_leaf_names()],
                                      dtype=int)
        self.set_params(alpha, theta, sigma2, regimes=regimes, root_value=root_value)

    def set_params(self, alpha, theta, sigma2, regimes=None, root_value=None):
        """Recompute only the rate-dependent edge quantities on the cached structure (vectorized)."""
        N = self.N
        thetas = np.atleast_1d(np.asarray(theta, dtype=float))
        theta_vec = (np.full(N, thetas[0]) if self._regime_idx is None
                     else thetas[self._regime_idx])
        # per-edge contraction phi and variance factor v (BM: phi=1, v=dist)
        if alpha is None or alpha <= 0:
            phi = np.ones(N); v = self.dist.copy()
        else:
            phi = np.exp(-alpha * self.dist)
            v = -np.expm1(-2.0 * alpha * self.dist) / (2.0 * alpha)
        V = v * sigma2
        self.phi = phi
        self.c = (1.0 - phi) * theta_vec              # 0 for BM
        self.free = V > self._ZERO
        nonroot = self._nonroot
        if np.any(V[nonroot] <= self._ZERO):
            raise ValueError("Zero-length non-root branch is not supported; "
                             "collapse or perturb such branches before fitting.")
        self.invV = np.where(self.free, 1.0 / np.where(V > 0, V, 1.0), 0.0)
        # prior mean of the root (the only possibly-fixed node)
        a = float(theta_vec[self.is_root][0]) if root_value is None \
            else float(np.asarray(root_value).ravel()[0])
        self.mu0 = np.zeros(N)
        self.mu0[self.is_root] = phi[self.is_root] * a + (1.0 - phi[self.is_root]) * theta_vec[self.is_root]
        # elimination couples a node only to a *free* parent
        self.solve_parent = np.full(N, -1, dtype=int)
        self.off = np.zeros(N)
        parent_free = np.zeros(N, dtype=bool)
        parent_free[nonroot] = self.free[self._nr_parent]
        couple = nonroot[parent_free[nonroot]]
        self.solve_parent[couple] = self.parent[couple]
        self.off[couple] = -self.phi[couple] * self.invV[couple]
        # constant part of the Hessian diagonal (prior precision)
        self.diag_base = np.where(self.free, self.invV, 0.0).copy()
        np.add.at(self.diag_base, self.parent[couple], self.phi[couple] ** 2 * self.invV[couple])
        self.log_det_Q = float(np.sum(np.log(self.invV[self.free])))
        self.fixed_value = {i: self.mu0[i] for i in np.where(~self.free)[0]}

    def prior_grad(self, z):
        """Gradient of the prior negative log-density wrt all free node latents."""
        g = np.zeros(self.N)
        rt = self.is_root & self.free
        g[rt] += (z[rt] - self.mu0[rt]) * self.invV[rt]
        nr = self._nonroot; pa = self._nr_parent
        r = (z[nr] - self.phi[nr] * z[pa] - self.c[nr]) * self.invV[nr]
        g[nr] += r
        np.add.at(g, pa, -self.phi[nr] * r)
        g[~self.free] = 0.0
        return g

    def prior_quad(self, z):
        """(z - m)^T Q (z - m): sum of per-edge prior quadratics."""
        rt = self.is_root & self.free
        total = np.sum((z[rt] - self.mu0[rt]) ** 2 * self.invV[rt])
        nr = self._nonroot; pa = self._nr_parent
        resid = z[nr] - self.phi[nr] * z[pa] - self.c[nr]
        total += np.sum(resid ** 2 * self.invV[nr])
        return float(total)

    def _eliminate(self, diag, rhs=None):
        """Postorder Gaussian elimination over free nodes; returns (d, rr, logdet)."""
        d = diag.copy()
        rr = None if rhs is None else rhs.copy()
        log_det = 0.0
        for nd in self.post:
            i = self.index[nd]
            if not self.free[i]:
                continue
            log_det += np.log(d[i])
            p = self.solve_parent[i]
            if p >= 0:
                o = self.off[i]
                d[p] -= o * o / d[i]
                if rr is not None:
                    rr[p] -= o * rr[i] / d[i]
        return d, rr, log_det

    def solve(self, diag, rhs):
        """Solve (Q + D) x = rhs over free nodes; return (x, log|Q+D|)."""
        d, rr, log_det = self._eliminate(diag, rhs)
        x = np.zeros(self.N)
        for nd in self.pre:
            i = self.index[nd]
            if not self.free[i]:
                continue
            p = self.solve_parent[i]
            if p < 0:
                x[i] = rr[i] / d[i]
            else:
                x[i] = (rr[i] - self.off[i] * x[p]) / d[i]
        return x, log_det

    def log_det(self, diag):
        return self._eliminate(diag, None)[2]


def latent_tree_laplace_marginal(tree, obs, alpha, theta, sigma2, regimes=None,
                                 root_value=None, max_iter=100, tol=1e-8, f_clip=40.0,
                                 model=None):
    """O(n) Laplace approximation to log p(y | hyperparameters) for a latent tree.

    Parameters mirror the OU model: scalar ``alpha`` (alpha<=0 => BM), optimum(s)
    ``theta`` ((p=1) scalar or per-regime array), diffusion variance ``sigma2``,
    optional ``regimes`` painting, and fixed ancestral ``root_value``.
    ``obs`` is the observation model (see module docstring).

    ``model`` optionally reuses a prebuilt :class:`_TreeModel` (the tree structure is fixed across
    an optimization; only the rate params change), avoiding a full rebuild on every evaluation.
    """
    if model is None:
        M = _TreeModel(tree, alpha, theta, sigma2, regimes=regimes, root_value=root_value)
    else:
        M = model
        M.set_params(alpha, theta, sigma2, regimes=regimes, root_value=root_value)
    leaf_idx = M.leaf_node_idx

    def psi(z):
        return -obs.loglik(z[leaf_idx]) + 0.5 * M.prior_quad(z)

    # Initialize: leaves near the data mode, internal nodes at the mean of those.
    z = np.zeros(M.N)
    leaf_init = obs.mode_init()
    z[:] = float(np.mean(leaf_init))
    z[leaf_idx] = leaf_init
    # Clip leaves to a sane range around their data-driven init.
    z[leaf_idx] = np.clip(z[leaf_idx], leaf_init - f_clip, leaf_init + f_clip)
    for i, val in M.fixed_value.items():       # pin fixed nodes (e.g. zero-length root)
        z[i] = val
    cur = psi(z)

    for _ in range(max_iter):
        f = z[leaf_idx]
        W = obs.neg_hess_diag(f)
        g_obs = obs.grad(f)

        grad = M.prior_grad(z)
        grad[leaf_idx] -= g_obs

        diag = M.diag_base.copy()
        diag[leaf_idx] += W

        step, _ = M.solve(diag, -grad)

        t = 1.0
        new = cur
        for _ in range(40):
            z_try = z + t * step
            new = psi(z_try)
            if new <= cur:
                break
            t *= 0.5
        converged = abs(new - cur) < tol
        z, cur = z_try, new
        if converged:
            break

    f = z[leaf_idx]
    W = obs.neg_hess_diag(f)
    diag = M.diag_base.copy()
    diag[leaf_idx] += W
    log_det_QW = M.log_det(diag)

    return obs.loglik(f) - 0.5 * M.prior_quad(z) + 0.5 * M.log_det_Q - 0.5 * log_det_QW
