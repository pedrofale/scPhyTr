"""Model 0 validation: the analytic OU tip distribution must match forward simulation.

The analytic mean/covariance (sparseou.ou) and the forward branch-by-branch simulator
(sparseou.simulate) are written independently, so agreement is a genuine correctness check.
"""
import numpy as np

from scphytr.modes._tree import Tree
from scphytr.modes._ou import node_optima, tip_mean, tip_cov, loglik
from scphytr.modes.simulate import simulate_tips, draw_events


def _setup(seed=0, n_leaves=8, alpha=0.9, sigma=0.8):
    tree = Tree.balanced(n_leaves, branch_length=0.4)
    rng = np.random.default_rng(seed)
    theta0 = rng.normal(size=3)
    _, _, delta, _ = draw_events(tree, 3, n_events=2, frac_genes=0.7, omega=1.5, rng=rng,
                                 min_clade=2)
    return tree, alpha, sigma, theta0, delta


def test_mean_matches_monte_carlo():
    tree, alpha, sigma, theta0, delta = _setup()
    th = node_optima(tree, theta0, delta)
    M = tip_mean(tree, alpha, th)
    rng = np.random.default_rng(1)
    S = np.stack([simulate_tips(tree, alpha, sigma, theta0, delta, rng=rng, root="stationary")
                  for _ in range(4000)])
    mc = S.mean(0)
    err = np.abs(mc - M).max()
    assert err < 0.08, f"analytic vs MC mean mismatch {err:.3f}"


def test_cov_matches_monte_carlo():
    tree, alpha, sigma, theta0, delta = _setup()
    V = tip_cov(tree, alpha, sigma, root="stationary")
    rng = np.random.default_rng(2)
    S = np.stack([simulate_tips(tree, alpha, sigma, theta0, delta, rng=rng, root="stationary")[:, 0]
                  for _ in range(20000)])
    mc = np.cov(S, rowvar=False)
    err = np.abs(mc - V).max() / V.max()
    assert err < 0.06, f"analytic vs MC covariance mismatch (rel) {err:.3f}"


def test_stationary_cov_is_patristic():
    """With a stationary root, Cov(i,j) = sigma^2/(2 alpha) * exp(-alpha * patristic distance)."""
    tree = Tree.balanced(8, branch_length=0.3)
    alpha, sigma = 1.3, 0.7
    V = tip_cov(tree, alpha, sigma, root="stationary")
    n = tree.n_leaves
    depth, parent = tree.depth, tree.parent

    def anc(v):
        out = []
        while v >= 0:
            out.append(v); v = parent[v]
        return out

    D = np.zeros((n, n))
    for i in range(n):
        ai = anc(tree.leaves[i])
        for j in range(n):
            aj = set(anc(tree.leaves[j]))
            mrca = next(a for a in ai if a in aj)
            D[i, j] = depth[tree.leaves[i]] + depth[tree.leaves[j]] - 2 * depth[mrca]
    expect = (sigma ** 2 / (2 * alpha)) * np.exp(-alpha * D)
    assert np.abs(V - expect).max() < 1e-10


def test_brownian_limit():
    """As alpha -> 0 with a fixed root, Cov -> sigma^2 * (shared root-path length)."""
    tree = Tree.balanced(8, branch_length=0.5)
    sigma = 1.1
    V = tip_cov(tree, 1e-12, sigma, root="fixed")
    depth, parent = tree.depth, tree.parent

    def anc(v):
        out = []
        while v >= 0:
            out.append(v); v = parent[v]
        return out

    n = tree.n_leaves
    expect = np.zeros((n, n))
    for i in range(n):
        ai = anc(tree.leaves[i])
        for j in range(n):
            aj = set(anc(tree.leaves[j]))
            mrca = next(a for a in ai if a in aj)
            expect[i, j] = sigma ** 2 * depth[mrca]
    assert np.abs(V - expect).max() < 1e-8


def test_loglik_matches_scipy_mvn():
    from scipy.stats import multivariate_normal
    tree, alpha, sigma, theta0, delta = _setup()
    th = node_optima(tree, theta0, delta)
    M = tip_mean(tree, alpha, th)
    V = tip_cov(tree, alpha, sigma, root="stationary")
    rng = np.random.default_rng(5)
    Y = simulate_tips(tree, alpha, sigma, theta0, delta, rng=rng)
    ll = loglik(Y, tree, alpha, sigma, theta0, delta, root="stationary")
    ref = sum(multivariate_normal(M[:, g], V, allow_singular=True).logpdf(Y[:, g])
              for g in range(Y.shape[1]))
    assert abs(ll - ref) < 1e-6, f"{ll} vs {ref}"


def test_loglik_peaks_near_truth():
    """Sanity: the exact likelihood prefers the true alpha over clearly wrong ones."""
    tree = Tree.balanced(64, branch_length=0.15)
    alpha_true, sigma = 1.5, 0.8
    rng = np.random.default_rng(7)
    theta0 = np.zeros(40)
    _, _, delta, _ = draw_events(tree, 40, n_events=2, frac_genes=0.3, omega=2.0, rng=rng)
    Y = simulate_tips(tree, alpha_true, sigma, theta0, delta, rng=rng)
    grid = [0.15, 0.5, 1.5, 4.5]
    lls = [loglik(Y, tree, a, sigma, theta0, delta) for a in grid]
    assert grid[int(np.argmax(lls))] == alpha_true, dict(zip(grid, np.round(lls, 1)))
