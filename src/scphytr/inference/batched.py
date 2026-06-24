"""Batched scalar tree-Laplace: per-gene BM/OU fits for many genes at once.

The per-gene scan is bottlenecked by Python loops over tree nodes, repeated for
every gene and every optimizer step. Those loops are identical across genes -- only
the per-node scalars differ -- so carrying a *gene axis* turns each per-node scalar
op into a vectorized ``(G,)`` op and the Python loop is paid once for all G genes.

This module fits, for G genes simultaneously, neutral BM, single-optimum OU-1, and
multi-regime OU-2 by maximizing the batched Laplace marginal over a grid of
``(alpha, sigma2)`` with the optimum ``theta`` profiled in closed form at the mode.

The leaf observation is pluggable (``LeafObs``): ``PoissonLeaf`` (pure-Poisson on
per-subclone summed counts, optionally quasi-Poisson) or ``NBLeaf`` (per-cell
negative-binomial, modelling within-subclone overdispersion / plasticity). The
batched marginal matches the per-gene ``tree_laplace.latent_tree_laplace_marginal``
to machine precision.
"""
import numpy as np
from scipy.special import gammaln


# --------------------------------------------------------------------------- #
# Leaf observations: map a leaf-latent f (n_leaves, G) to (grad, W, loglik).
# --------------------------------------------------------------------------- #
class PoissonLeaf:
    """Pure-Poisson on per-subclone *summed* counts (optionally quasi-Poisson).

    Y, S : (n_leaves, G) summed counts and offsets. ``disp`` (G,) or scalar >= 1
    is a quasi-Poisson dispersion: the score and information are scaled by 1/disp.
    """

    def __init__(self, Y, S, disp=1.0):
        self.Y = np.asarray(Y, float)
        self.S = np.asarray(S, float)
        self.disp = disp
        self.nL, self.G = self.Y.shape

    def init_leaf(self):
        return np.log((self.Y + 0.5) / (self.S + 1e-9))

    def terms(self, f):
        rate = self.S * np.exp(f)
        grad = (self.Y - rate) / self.disp
        W = rate / self.disp
        ll = np.sum((self.Y * f - rate) / self.disp - gammaln(self.Y + 1.0), axis=0)
        return grad, W, ll


class NBLeaf:
    """Per-cell negative-binomial: cells within a subclone share the leaf latent.

    Y_i ~ NB(mean = S_i e^{f_leaf(i)}, size r_g). The within-subclone overdispersion
    ``r`` models plasticity beyond Poisson shot noise. Cells are reduced to leaves
    by a (n_leaves x n_cells) incidence matrix (fast BLAS, not np.add.at).

    counts : (n_cells, G); offsets : (n_cells,); leaf_index : (n_cells,);
    r : (G,) NB size per gene (smaller = more overdispersed).
    """

    def __init__(self, counts, offsets, leaf_index, n_leaves, r):
        self.y = np.asarray(counts, float)                  # (n_cells, G)
        self.s = np.asarray(offsets, float)[:, None]        # (n_cells, 1)
        self.idx = np.asarray(leaf_index, int)
        self.nL = int(n_leaves)
        self.G = self.y.shape[1]
        self.r = np.asarray(r, float)[None, :]              # (1, G)
        nc = self.y.shape[0]
        M = np.zeros((self.nL, nc))                         # incidence (leaf x cell)
        M[self.idx, np.arange(nc)] = 1.0
        self.M = M
        # per-leaf summed counts/offsets for init
        self.Ytot = M @ self.y
        self.Stot = M @ (self.s * np.ones((1, self.G)))

    @staticmethod
    def estimate_r(counts, offsets, leaf_index, n_leaves, r_max=1e4):
        """Moment estimate of NB size r per gene from within-subclone residuals.

        Pearson = sum (y-mu)^2/mu with mu the per-subclone Poisson mean; under NB
        E[Pearson] = n_cells + (sum mu)/r, so r = (sum mu)/(Pearson - n_cells).
        """
        y = np.asarray(counts, float); s = np.asarray(offsets, float)[:, None]
        idx = np.asarray(leaf_index, int); nc = y.shape[0]
        M = np.zeros((n_leaves, nc)); M[idx, np.arange(nc)] = 1.0
        Ytot = M @ y; Stot = M @ (s * np.ones((1, y.shape[1])))
        mu = s * (Ytot / np.maximum(Stot, 1e-9))[idx]       # (n_cells, G)
        pearson = np.sum((y - mu) ** 2 / np.maximum(mu, 1e-9), axis=0)
        summu = mu.sum(0)
        denom = np.maximum(pearson - nc, 1e-6)
        return np.clip(summu / denom, 1e-2, r_max)

    def init_leaf(self):
        return np.log((self.Ytot + 0.5) / (self.Stot + 1e-9))

    def terms(self, f):
        mu = self.s * np.exp(f[self.idx])                   # (n_cells, G)
        r = self.r
        g_cell = self.y - (self.y + r) * mu / (r + mu)
        W_cell = (self.y + r) * mu * r / (r + mu) ** 2
        grad = self.M @ g_cell
        W = self.M @ W_cell
        ll = np.sum(gammaln(self.y + r) - gammaln(r) - gammaln(self.y + 1.0)
                    + r * np.log(r / (r + mu)) + self.y * np.log(mu / (r + mu)), axis=0)
        return grad, W, ll


class BatchedTreeLaplace:
    """Topology + batched BM/OU Laplace marginal and grid fitter for G genes."""

    _ZERO = 1e-12

    def __init__(self, tree, regimes=None):
        post = list(tree.root.traverse("postorder"))
        index = {nd: i for i, nd in enumerate(post)}
        N = len(post)
        self.N = N
        self.parent = np.full(N, -1, dtype=int)
        self.dist = np.zeros(N)
        for nd in post:
            i = index[nd]
            self.dist[i] = nd.dist
            if nd.up is not None:
                self.parent[i] = index[nd.up]
        self.order = list(range(N))                      # already postorder
        self.is_root = self.parent < 0
        self.root_i = int(np.where(self.is_root)[0][0])
        name_to_i = {nd.name: index[nd] for nd in post if nd.is_leaf()}
        self.leaf_order = np.array([name_to_i[n] for n in tree.phylotree.get_leaf_names()])
        if regimes is None:
            self.regime_of = np.zeros(N, dtype=int)
            self.n_reg = 1
        else:
            self.regime_of = np.array([regimes[nd] for nd in post], dtype=int)
            self.n_reg = int(self.regime_of.max()) + 1

    def _branch(self, alpha):
        if alpha is None or alpha <= 0:
            return np.ones(self.N), self.dist.copy()
        phi = np.exp(-alpha * self.dist)
        v = -np.expm1(-2.0 * alpha * self.dist) / (2.0 * alpha)
        return phi, v

    def _precompute(self, alpha, sigma2):
        phi, v = self._branch(alpha)
        v = np.maximum(v, self._ZERO)
        invV = (1.0 / v)[:, None] / sigma2[None, :]      # (N, G)
        return phi, invV

    def _mode(self, obs, phi, invV, theta_reg, n_newton=30, tol=1e-9):
        """Batched Newton posterior mode of the latent Z (N, G)."""
        N, G = invV.shape
        parent, order, leaf = self.parent, self.order, self.leaf_order
        c = (1.0 - phi)[:, None] * theta_reg
        Z = np.array(theta_reg)
        Z[leaf] = obs.init_leaf()
        for _ in range(n_newton):
            gl, Wl, _ = obs.terms(Z[leaf])
            g = np.zeros((N, G)); g[leaf] = gl
            W = np.zeros((N, G)); W[leaf] = Wl
            for i in order:                              # prior gradient
                pa = parent[i]
                if pa >= 0:
                    r = invV[i] * (Z[i] - phi[i] * Z[pa] - c[i])
                    g[i] -= r
                    g[pa] += phi[i] * r
                else:
                    g[i] -= invV[i] * (Z[i] - theta_reg[i])
            step, _ = self._solve(phi, invV, W, g)
            Z = Z + step
            if np.max(np.abs(step)) < tol:
                break
        return Z

    def _solve(self, phi, invV, W, rhs):
        N, G = invV.shape
        parent, order = self.parent, self.order
        d = np.zeros((N, G))
        for i in order:
            d[i] += invV[i]
            pa = parent[i]
            if pa >= 0:
                d[pa] += phi[i] ** 2 * invV[i]
        d = d + W
        rr = rhs.copy()
        logdet = np.zeros(G)
        for i in order:
            pa = parent[i]
            logdet += np.log(d[i])
            if pa >= 0:
                off = -phi[i] * invV[i]
                d[pa] -= off * off / d[i]
                rr[pa] -= off * rr[i] / d[i]
        x = np.zeros((N, G))
        for i in reversed(order):
            pa = parent[i]
            if pa < 0:
                x[i] = rr[i] / d[i]
            else:
                off = -phi[i] * invV[i]
                x[i] = (rr[i] - off * x[pa]) / d[i]
        return x, logdet

    def _theta_at_mode(self, Z, phi, invV):
        N, G = invV.shape
        parent, order = self.parent, self.order
        num = np.zeros((self.n_reg, G)); den = np.zeros((self.n_reg, G))
        for i in order:
            r = self.regime_of[i]
            pa = parent[i]
            if pa >= 0:
                w = invV[i] * (1.0 - phi[i])
                num[r] += w * (Z[i] - phi[i] * Z[pa])
                den[r] += w * (1.0 - phi[i])
            else:
                num[r] += invV[i] * Z[i]
                den[r] += invV[i]
        return num / np.maximum(den, self._ZERO)

    def marginal(self, obs, alpha, sigma2, theta, n_inner=3):
        """Batched Laplace marginal log-lik (G,); profiles theta. Returns (logml, theta)."""
        phi, invV = self._precompute(alpha, sigma2)
        theta = np.array(theta, dtype=float)
        for _ in range(n_inner):
            Z = self._mode(obs, phi, invV, theta[self.regime_of])
            theta = self._theta_at_mode(Z, phi, invV)
        theta_reg = theta[self.regime_of]
        Z = self._mode(obs, phi, invV, theta_reg)
        leaf = self.leaf_order
        _, Wl, loglik = obs.terms(Z[leaf])
        quad = np.zeros(obs.G); logdetQ = np.zeros(obs.G)
        c = (1.0 - phi)[:, None] * theta_reg
        for i in self.order:
            pa = self.parent[i]
            if pa >= 0:
                quad += invV[i] * (Z[i] - phi[i] * Z[pa] - c[i]) ** 2
            else:
                quad += invV[i] * (Z[i] - theta[self.regime_of[i]]) ** 2
            logdetQ += np.log(invV[i])
        W = np.zeros((self.N, obs.G)); W[leaf] = Wl
        _, logdetH = self._solve(phi, invV, W, np.zeros((self.N, obs.G)))
        return loglik - 0.5 * quad + 0.5 * logdetQ - 0.5 * logdetH, theta

    def fit(self, obs, alpha_grid, sigma2_grid):
        """Grid-maximize the batched marginal over (alpha, sigma2); profile theta."""
        G = obs.G
        theta0 = np.tile(obs.init_leaf().mean(0), (self.n_reg, 1))
        best_ml = np.full(G, -np.inf)
        best = dict(alpha=np.zeros(G), sigma2=np.ones(G), theta=theta0.copy())
        for a in alpha_grid:
            for s2 in sigma2_grid:
                ml, theta = self.marginal(obs, a, np.full(G, s2), theta0)
                win = ml > best_ml
                best_ml = np.where(win, ml, best_ml)
                best["alpha"] = np.where(win, a, best["alpha"])
                best["sigma2"] = np.where(win, s2, best["sigma2"])
                best["theta"] = np.where(win[None, :], theta, best["theta"])
        best["logml"] = best_ml
        return best
