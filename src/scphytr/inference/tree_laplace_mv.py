"""O(n p^3) Laplace marginal for a *multivariate* latent tree model.

Each node carries a p-vector latent ``z_u`` (e.g. log-expression of p genes)
evolving along edges by a multivariate BM/OU transition with full diffusion
matrix ``K`` (p x p) -- whose off-diagonals are the evolutionary covariances
between traits. The p genes are observed at the leaves through *any* per-leaf
likelihood (Gaussian, Poisson counts, ...) that factorizes over genes given the
latent.

The latent prior precision is Kronecker, ``Q = A ⊗ K^{-1}`` (``A`` the scalar
tree precision, ``K^{-1}`` the trait precision), so the Laplace posterior
precision ``Q + W`` (``W`` the block-diagonal observation curvature) is
block-tree-structured: each node's p-block couples only to its parent's. Block
Gaussian elimination in post-order -- the multivariate generalization of
Felsenstein's pruning -- yields the posterior mode and ``log|Q + W|`` in
O(n p^3), never forming a dense covariance.

The estimated parameters (``alpha``, ``theta``, ``K``) live entirely in the
latent BM/OU model; the observation model only has to expose, on the leaf-latent
matrix ``F`` of shape (n_leaves, p):
    loglik(F) -> float, grad(F) -> (n,p), neg_hess_diag(F) -> (n,p) >= 0,
    mode_init() -> (n,p).
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from .tree_laplace import _ou_branch


class _MVTreeModel:
    """Precomputed multivariate BM/OU latent-tree Gaussian (precision A ⊗ K^{-1})."""

    _ZERO = 1e-12

    def __init__(self, tree, alpha, theta, K, regimes=None, root_value=None):
        thetas = np.atleast_2d(np.asarray(theta, dtype=float))   # (n_regimes, p)
        p = thetas.shape[1]
        self.p = p
        is_bm = alpha is None or alpha <= 0

        def theta_of(node):
            return thetas[0] if regimes is None else thetas[regimes[node]]

        root = tree.root
        self.post = list(root.traverse("postorder"))
        self.pre = list(root.traverse("preorder"))
        self.index = {nd: i for i, nd in enumerate(self.post)}
        N = len(self.post)
        self.N = N

        self.parent = np.full(N, -1, dtype=int)
        self.phi = np.zeros(N)
        self.invV = np.zeros(N)
        self.c = np.zeros((N, p))
        self.is_root = np.zeros(N, dtype=bool)
        self.free = np.ones(N, dtype=bool)
        self.mu0 = np.zeros((N, p))

        a = (theta_of(root) if root_value is None
             else np.asarray(root_value, dtype=float).ravel())

        for nd in self.post:
            i = self.index[nd]
            phi, v = (1.0, nd.dist) if is_bm else _ou_branch(alpha, nd.dist)
            if nd is root:
                self.is_root[i] = True
                self.mu0[i] = phi * a + (1.0 - phi) * theta_of(nd)
                if v <= self._ZERO:
                    self.free[i] = False
                else:
                    self.invV[i] = 1.0 / v
            else:
                if v <= self._ZERO:
                    raise ValueError("Zero-length non-root branch is not supported.")
                self.parent[i] = self.index[nd.up]
                self.phi[i] = phi
                self.invV[i] = 1.0 / v
                self.c[i] = (1.0 - phi) * theta_of(nd)

        self.solve_parent = np.full(N, -1, dtype=int)
        self.off = np.zeros(N)
        for i in range(N):
            pa = self.parent[i]
            if pa >= 0 and self.free[pa]:
                self.solve_parent[i] = pa
                self.off[i] = -self.phi[i] * self.invV[i]

        # Scalar coefficient of K^{-1} on each block diagonal (prior part).
        self.diag_coef = np.zeros(N)
        for i in range(N):
            if self.free[i]:
                self.diag_coef[i] += self.invV[i]
            pa = self.parent[i]
            if pa >= 0 and self.free[pa]:
                self.diag_coef[pa] += self.phi[i] ** 2 * self.invV[i]

        self.K = np.asarray(K, dtype=float)
        self.P = np.linalg.inv(self.K)
        sign, logdetK = np.linalg.slogdet(self.K)
        if sign <= 0:
            raise ValueError("Trait covariance K must be positive definite.")
        n_free = int(self.free.sum())
        log_det_A = float(np.sum(np.log(self.invV[self.free])))
        # log|Q| = log|A ⊗ K^{-1}| = p log|A| - n_free log|K|.
        self.log_det_Q = p * log_det_A - n_free * logdetK

        node_by_name = {nd.name: self.index[nd] for nd in self.post if nd.is_leaf()}
        self.leaf_node_idx = np.array([node_by_name[name] for name in tree.phylotree.get_leaf_names()],
                                      dtype=int)
        self.fixed = [i for i in range(N) if not self.free[i]]

        # Static geometry used by the EM M-step (branch length and regime per node).
        self.t = np.array([nd.dist for nd in self.post], dtype=float)
        self.regime_idx = np.array([0 if regimes is None else regimes[nd] for nd in self.post],
                                   dtype=int)
        self.root_idx = self.index[root]
        self.root_regime = 0 if regimes is None else regimes[root]
        self.n_regimes = thetas.shape[0]

    def prior_grad(self, Z):
        """(A ⊗ K^{-1})(Z - m) as an (N, p) array (gradient of prior nll)."""
        P = self.P
        G = np.zeros((self.N, self.p))
        for i in range(self.N):
            if self.is_root[i]:
                if self.free[i]:
                    G[i] += self.invV[i] * (P @ (Z[i] - self.mu0[i]))
            else:
                pa = self.parent[i]
                d = Z[i] - self.phi[i] * Z[pa] - self.c[i]
                r = self.invV[i] * (P @ d)
                G[i] += r
                G[pa] += -self.phi[i] * r
        for i in self.fixed:
            G[i] = 0.0
        return G

    def prior_quad(self, Z):
        """(Z - m)^T (A ⊗ K^{-1}) (Z - m)."""
        P = self.P
        total = 0.0
        for i in range(self.N):
            if self.is_root[i]:
                if self.free[i]:
                    d = Z[i] - self.mu0[i]
                    total += self.invV[i] * float(d @ P @ d)
            else:
                pa = self.parent[i]
                d = Z[i] - self.phi[i] * Z[pa] - self.c[i]
                total += self.invV[i] * float(d @ P @ d)
        return total

    def _eliminate(self, Wdiag, rhs=None):
        """Block postorder elimination of (Q + diag(W)); returns (cho, rr, logdet)."""
        P = self.P
        D = {}
        cho = {}
        rr = None if rhs is None else rhs.copy()
        log_det = 0.0
        for nd in self.post:
            i = self.index[nd]
            if not self.free[i]:
                continue
            if i not in D:
                D[i] = self.diag_coef[i] * P + np.diag(Wdiag[i])
            ch = cho_factor(D[i], lower=True)
            cho[i] = ch
            log_det += 2.0 * float(np.sum(np.log(np.diag(ch[0]))))
            pp = self.solve_parent[i]
            if pp >= 0:
                o = self.off[i]
                if pp not in D:
                    D[pp] = self.diag_coef[pp] * P + np.diag(Wdiag[pp])
                # Schur complement: -o^2 P (D_i)^{-1} P  (off block = o P).
                D[pp] = D[pp] - o * o * (P @ cho_solve(ch, P))
                if rr is not None:
                    rr[pp] = rr[pp] - o * (P @ cho_solve(ch, rr[i]))
        return cho, rr, log_det

    def solve(self, Wdiag, rhs):
        """Solve (Q + diag(W)) X = rhs (both (N, p)); return (X, log|Q+W|)."""
        cho, rr, log_det = self._eliminate(Wdiag, rhs)
        P = self.P
        X = np.zeros((self.N, self.p))
        for nd in self.pre:
            i = self.index[nd]
            if not self.free[i]:
                continue
            pp = self.solve_parent[i]
            if pp < 0:
                X[i] = cho_solve(cho[i], rr[i])
            else:
                X[i] = cho_solve(cho[i], rr[i] - self.off[i] * (P @ X[pp]))
        return X, log_det

    def log_det(self, Wdiag):
        return self._eliminate(Wdiag, None)[2]

    def posterior_covariances(self, Wdiag):
        """Posterior covariance blocks of (Q + diag(W))^{-1} via a tree smoother.

        Returns (Sigma, cross): ``Sigma[i]`` is Cov(z_i) (p x p) and ``cross[i]``
        is Cov(z_i, z_parent(i)) for a free parent (zeros otherwise). Reuses the
        Cholesky factors of the eliminated pivots (Rauch-Tung-Striebel recursion
        on the tree), so it is O(n p^3).
        """
        cho, _, _ = self._eliminate(Wdiag, None)
        P = self.P
        p = self.p
        eye = np.eye(p)
        Sigma = np.zeros((self.N, p, p))
        cross = np.zeros((self.N, p, p))
        for nd in self.pre:                      # parents before children
            i = self.index[nd]
            if not self.free[i]:
                continue
            d_inv = cho_solve(cho[i], eye)
            pp = self.solve_parent[i]
            if pp < 0:
                Sigma[i] = d_inv
            else:
                G = -self.off[i] * cho_solve(cho[i], P)   # -d_i^{-1} (o_i P)
                S_pa = Sigma[pp]
                cross[i] = G @ S_pa
                Sigma[i] = d_inv + G @ S_pa @ G.T
        return Sigma, cross

    def sample_gaussian(self, Wdiag, mean, n_samples, rng):
        """Draw exact samples from ``N(mean, (Q + diag(W))^{-1})`` in O(n p^3).

        Forward-filter/backward-sample (simulation smoother) reusing the same
        Cholesky pivots and gains ``G_i = -d_i^{-1} o_i P`` as
        :meth:`posterior_covariances`. In centred coordinates the root is drawn
        from ``N(0, d_root^{-1})`` and each child as ``z_i = G_i z_pa + eps_i``
        with ``eps_i ~ N(0, d_i^{-1})``; ``mean`` is added back at the end. The
        resulting draws have covariance exactly equal to the smoother blocks.

        Returns an array of shape ``(n_samples, N, p)``.
        """
        cho, _, _ = self._eliminate(Wdiag, None)
        P = self.P
        p = self.p
        delta = np.zeros((n_samples, self.N, p))
        for nd in self.pre:                      # parents before children
            i = self.index[nd]
            if not self.free[i]:
                continue
            Lf = cho[i][0]
            # eps ~ N(0, d_i^{-1}): solve L^T eps = z  =>  Cov(eps) = (L L^T)^{-1}.
            z = rng.standard_normal((p, n_samples))
            eps = solve_triangular(Lf, z, lower=True, trans="T").T   # (n_samples, p)
            pp = self.solve_parent[i]
            if pp < 0:
                delta[:, i, :] = eps
            else:
                G = -self.off[i] * cho_solve(cho[i], P)              # (p, p)
                delta[:, i, :] = delta[:, pp, :] @ G.T + eps
        out = mean[None, :, :] + delta
        for i in self.fixed:
            out[:, i, :] = mean[i]
        return out


def _newton_mode(M, obs, max_iter=100, tol=1e-8):
    """Laplace posterior mode of the latent Z (N, p) via damped Newton."""
    leaf_idx = M.leaf_node_idx
    p = M.p

    def psi(Z):
        return -obs.loglik(Z[leaf_idx]) + 0.5 * M.prior_quad(Z)

    leaf_init = obs.mode_init()
    Z = np.zeros((M.N, p))
    Z[:] = leaf_init.mean(axis=0)
    Z[leaf_idx] = leaf_init
    for i in M.fixed:
        Z[i] = M.mu0[i]
    cur = psi(Z)

    for _ in range(max_iter):
        F = Z[leaf_idx]
        grad = M.prior_grad(Z)
        grad[leaf_idx] -= obs.grad(F)
        Wdiag = np.zeros((M.N, p))
        Wdiag[leaf_idx] = obs.neg_hess_diag(F)

        step, _ = M.solve(Wdiag, -grad)

        t = 1.0
        new = cur
        for _ in range(40):
            Z_try = Z + t * step
            new = psi(Z_try)
            if new <= cur:
                break
            t *= 0.5
        converged = abs(new - cur) < tol
        Z, cur = Z_try, new
        if converged:
            break
    return Z


def mv_tree_laplace_marginal(tree, obs, alpha, theta, K, regimes=None, root_value=None,
                             max_iter=100, tol=1e-8):
    """O(n p^3) Laplace marginal log p(Y | alpha, theta, K) for a multivariate latent tree.

    ``theta`` is the OU optimum: shape (p,) for a single regime or (n_regimes, p)
    with a ``regimes`` painting. ``alpha <= 0`` selects multivariate BM. ``K`` is
    the p x p diffusion (rate) matrix. ``obs`` is the multivariate observation
    model (see module docstring).
    """
    M = _MVTreeModel(tree, alpha, theta, K, regimes=regimes, root_value=root_value)
    Z = _newton_mode(M, obs, max_iter=max_iter, tol=tol)

    F = Z[M.leaf_node_idx]
    Wdiag = np.zeros((M.N, M.p))
    Wdiag[M.leaf_node_idx] = obs.neg_hess_diag(F)
    log_det_QW = M.log_det(Wdiag)

    return obs.loglik(F) - 0.5 * M.prior_quad(Z) + 0.5 * M.log_det_Q - 0.5 * log_det_QW


def mv_laplace_estep(tree, obs, alpha, theta, K, regimes=None, root_value=None,
                     max_iter=100, tol=1e-8):
    """E-step: Laplace posterior mode + covariance blocks for the latent tree.

    Returns a dict with the fitted ``_MVTreeModel`` ``M``, the posterior mode
    ``Z`` (N, p), and the posterior covariance blocks ``Sigma`` (N, p, p) and
    parent-child cross-covariances ``cross`` (N, p, p). The root must be a free
    latent (positive root branch).
    """
    M = _MVTreeModel(tree, alpha, theta, K, regimes=regimes, root_value=root_value)
    if not M.free[M.root_idx]:
        raise ValueError("EM requires a free root (positive root branch length).")
    Z = _newton_mode(M, obs, max_iter=max_iter, tol=tol)
    Wdiag = np.zeros((M.N, M.p))
    Wdiag[M.leaf_node_idx] = obs.neg_hess_diag(Z[M.leaf_node_idx])
    Sigma, cross = M.posterior_covariances(Wdiag)
    return {"M": M, "Z": Z, "Sigma": Sigma, "cross": cross}
