"""Validate the linear-time BM pruning log-likelihood against a dense oracle.

Asserts that ``bm_pruning_logpdf`` (O(n), no dense covariance) matches a direct
dense ``K ⊗ C`` multivariate-normal evaluation to numerical tolerance, across
random trait covariances/means and several tree shapes -- including a large tree
where the dense version is expensive but pruning stays cheap.

The dense reference covariance is built independently here (shared root-to-MRCA
path lengths) so the check does not depend on ``Tree.make_species_cov_matrix``,
which currently has a bug on deeper/unbalanced trees.
"""

import time
import numpy as np
import ete3
from scipy.stats import multivariate_normal

import scphytr
from scphytr.utils.pruning import bm_pruning_logpdf, ou_pruning_logpdf, paint_regimes


def build_tree(nwk_txt):
    tree = scphytr.utils.Tree()
    tree.phylotree = ete3.PhyloTree(nwk_txt, format=1)
    tree.root = tree.phylotree.get_tree_root()
    return tree


def reference_covariance(tree):
    """Dense n x n phylogenetic covariance: shared path length from the ancestral
    state (above the root) down to each pair's MRCA. Matches the BM convention
    where ``root.dist`` is the branch from the fixed ancestral state to the MRCA.
    """
    root = tree.root
    leaves = root.get_leaves()
    n = len(leaves)
    # Distance from the root node down to every node (root.dist excluded here).
    depth = {root: 0.0}
    for node in root.traverse("preorder"):
        for child in node.children:
            depth[child] = depth[node] + child.dist

    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mrca = leaves[i] if i == j else tree.phylotree.get_common_ancestor(leaves[i], leaves[j])
            C[i, j] = root.dist + (depth[leaves[i]] if i == j else depth[mrca])
    return C, [leaf.name for leaf in leaves]


def dense_logpdf(tree, leaf_values, means, K):
    """log N(y; repeat(means, n), K ⊗ C) in trait-major order."""
    C, order = reference_covariance(tree)
    X = np.array([leaf_values[name] for name in order])  # n x p
    y = X.reshape(-1, order="F")                          # trait-major
    a = np.repeat(means, X.shape[0])
    V = np.kron(K, C)
    return multivariate_normal.logpdf(y, a, V)


def node_depths(tree):
    """Distance from the root node down to every node (root.dist excluded)."""
    root = tree.root
    depth = {root: 0.0}
    for node in root.traverse("preorder"):
        for child in node.children:
            depth[child] = depth[node] + child.dist
    return depth


def dense_logpdf_ou(tree, leaf_values, alpha, theta, K, root_value):
    """log N(y; mean_OU, K ⊗ C_OU) for multivariate OU with scalar alpha.

    T_i = root-to-tip time (incl. root.dist), s_ij = root-to-MRCA time.
    C_OU[i,j] = e^{-alpha(T_i+T_j-2 s_ij)} (1 - e^{-2 alpha s_ij}) / (2 alpha)
    mean_i    = theta + e^{-alpha T_i} (root_value - theta)
    """
    root = tree.root
    leaves = root.get_leaves()
    n = len(leaves)
    depth = node_depths(tree)
    T = np.array([root.dist + depth[leaf] for leaf in leaves])

    s = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mrca = leaves[i] if i == j else tree.phylotree.get_common_ancestor(leaves[i], leaves[j])
            s[i, j] = root.dist + (depth[leaves[i]] if i == j else depth[mrca])

    C = np.exp(-alpha * (T[:, None] + T[None, :] - 2.0 * s)) * (1.0 - np.exp(-2.0 * alpha * s)) / (2.0 * alpha)

    p = theta.shape[0]
    order = [leaf.name for leaf in leaves]
    X = np.array([leaf_values[name] for name in order])  # n x p
    y = X.reshape(-1, order="F")
    a = np.asarray(root_value, dtype=float).ravel()
    # Trait-major mean: for trait r, leaf i -> theta_r + e^{-alpha T_i}(a_r - theta_r)
    mean = np.concatenate([theta[r] + np.exp(-alpha * T) * (a[r] - theta[r]) for r in range(p)])
    V = np.kron(K, C)
    return multivariate_normal.logpdf(y, mean, V)


def dense_logpdf_ou_regimes(tree, leaf_values, alpha, thetas, K, regimes, root_value):
    """Dense OU oracle with regime-specific optima (shared alpha).

    The covariance C_OU is unchanged by the optima; only the mean becomes
    regime-dependent. The per-node mean follows the deterministic OU recursion
    E[child] = e^{-alpha t} E[parent] + (1 - e^{-alpha t}) theta_{regime(child)},
    seeded above the root by the fixed ancestral state ``root_value``.
    """
    root = tree.root
    leaves = root.get_leaves()
    n = len(leaves)
    depth = node_depths(tree)
    T = np.array([root.dist + depth[leaf] for leaf in leaves])

    s = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mrca = leaves[i] if i == j else tree.phylotree.get_common_ancestor(leaves[i], leaves[j])
            s[i, j] = root.dist + (depth[leaves[i]] if i == j else depth[mrca])
    C = np.exp(-alpha * (T[:, None] + T[None, :] - 2.0 * s)) * (1.0 - np.exp(-2.0 * alpha * s)) / (2.0 * alpha)

    thetas = np.atleast_2d(thetas)
    a = np.asarray(root_value, dtype=float).ravel()
    E = {}
    phi_r = np.exp(-alpha * root.dist)
    E[root] = phi_r * a + (1.0 - phi_r) * thetas[regimes[root]]
    for node in root.traverse("preorder"):
        for child in node.children:
            phi = np.exp(-alpha * child.dist)
            E[child] = phi * E[node] + (1.0 - phi) * thetas[regimes[child]]

    p = thetas.shape[1]
    order = [leaf.name for leaf in leaves]
    X = np.array([leaf_values[name] for name in order])
    y = X.reshape(-1, order="F")
    mean = np.concatenate([np.array([E[leaf][r] for leaf in leaves]) for r in range(p)])
    V = np.kron(K, C)
    return multivariate_normal.logpdf(y, mean, V)


def random_spd(p, rng):
    A = rng.standard_normal((p, p))
    return A @ A.T + p * np.eye(p)


def assign_random_traits(tree, p, rng):
    leaf_values = {}
    species_trait_values = {}
    for leaf in tree.root.get_leaves():
        vals = rng.standard_normal(p)
        leaf_values[leaf.name] = vals
        species_trait_values[leaf.name] = {str(t): float(vals[t]) for t in range(p)}
    tree.set_trait_values(species_trait_values)
    return leaf_values


def check_tree(name, nwk_txt, p=3, n_trials=5, seed=0):
    rng = np.random.default_rng(seed)
    tree = build_tree(nwk_txt)
    n = len(tree.root.get_leaves())

    max_abs = 0.0
    for _ in range(n_trials):
        means = rng.standard_normal(p)
        K = random_spd(p, rng)
        leaf_values = assign_random_traits(tree, p, rng)

        dense = dense_logpdf(tree, leaf_values, means, K)
        pruned = bm_pruning_logpdf(tree, means, K)
        max_abs = max(max_abs, abs(dense - pruned))

    print(f"[BM {name}] n={n}, p={p}: max|dense - pruned| over {n_trials} trials = {max_abs:.3e}")
    return max_abs


def check_tree_ou(name, nwk_txt, p=3, n_trials=5, seed=0):
    rng = np.random.default_rng(seed)
    tree = build_tree(nwk_txt)
    n = len(tree.root.get_leaves())

    max_abs = 0.0
    for _ in range(n_trials):
        alpha = float(rng.uniform(0.2, 2.0))
        theta = rng.standard_normal(p)
        root_value = rng.standard_normal(p)
        K = random_spd(p, rng)
        leaf_values = assign_random_traits(tree, p, rng)

        dense = dense_logpdf_ou(tree, leaf_values, alpha, theta, K, root_value)
        pruned = ou_pruning_logpdf(tree, alpha, theta, K, root_value=root_value)
        max_abs = max(max_abs, abs(dense - pruned))

    print(f"[OU {name}] n={n}, p={p}: max|dense - pruned| over {n_trials} trials = {max_abs:.3e}")
    return max_abs


def check_ou_regimes(name, nwk_txt, shift_leaves, p=2, n_trials=5, seed=0):
    """Validate multi-regime OU pruning against the dense regime oracle, and that
    the single-regime painting reproduces plain OU-1."""
    rng = np.random.default_rng(seed)
    tree = build_tree(nwk_txt)
    n = len(tree.root.get_leaves())

    shift_node = tree.phylotree.get_common_ancestor(shift_leaves)
    regimes, n_regimes = paint_regimes(tree, [shift_node])
    all_zero = {node: 0 for node in tree.root.traverse()}

    max_abs = 0.0
    max_reduction = 0.0
    for _ in range(n_trials):
        alpha = float(rng.uniform(0.2, 2.0))
        thetas = rng.standard_normal((n_regimes, p))
        root_value = rng.standard_normal(p)
        K = random_spd(p, rng)
        leaf_values = assign_random_traits(tree, p, rng)

        dense = dense_logpdf_ou_regimes(tree, leaf_values, alpha, thetas, K, regimes, root_value)
        pruned = ou_pruning_logpdf(tree, alpha, thetas, K, regimes=regimes, root_value=root_value)
        max_abs = max(max_abs, abs(dense - pruned))

        # Single regime (all nodes regime 0) must equal plain OU-1.
        theta0 = thetas[0]
        ou1 = ou_pruning_logpdf(tree, alpha, theta0, K, root_value=theta0)
        ou1_reg = ou_pruning_logpdf(tree, alpha, theta0[None, :], K,
                                    regimes=all_zero, root_value=theta0)
        max_reduction = max(max_reduction, abs(ou1 - ou1_reg))

    print(f"[OU{n_regimes} {name}] n={n}, p={p}: max|dense - pruned| = {max_abs:.3e} | "
          f"OU-1 reduction = {max_reduction:.3e}")
    return max(max_abs, max_reduction)


def check_bm_limit(nwk_txt, p=3, seed=0):
    """As alpha -> 0, OU pruning should approach BM pruning (means = theta)."""
    rng = np.random.default_rng(seed)
    tree = build_tree(nwk_txt)
    theta = rng.standard_normal(p)
    K = random_spd(p, rng)
    assign_random_traits(tree, p, rng)

    bm = bm_pruning_logpdf(tree, theta, K)
    ou_small = ou_pruning_logpdf(tree, 1e-7, theta, K, root_value=theta)
    err = abs(bm - ou_small)
    print(f"[BM-limit] |BM - OU(alpha=1e-7)| = {err:.3e}")
    return err


def balanced_newick(depth):
    counter = [0]

    def rec(d, is_root):
        if d == 0:
            counter[0] += 1
            return f"L{counter[0]}:1.0"
        node = f"({rec(d - 1, False)},{rec(d - 1, False)})"
        return node if is_root else node + ":1.0"

    return rec(depth, True) + ";"


def timing_demo(depth=9, p=4, seed=0):
    rng = np.random.default_rng(seed)
    tree = build_tree(balanced_newick(depth))
    n = len(tree.root.get_leaves())
    means = rng.standard_normal(p)
    K = random_spd(p, rng)
    leaf_values = assign_random_traits(tree, p, rng)

    t0 = time.perf_counter()
    dense = dense_logpdf(tree, leaf_values, means, K)
    t1 = time.perf_counter()
    pruned = bm_pruning_logpdf(tree, means, K)
    t2 = time.perf_counter()

    print(
        f"[large] n={n}, p={p}: |dense - pruned|={abs(dense - pruned):.3e} | "
        f"dense {1e3 * (t1 - t0):.1f} ms, pruning {1e3 * (t2 - t1):.1f} ms"
    )
    return abs(dense - pruned)


if __name__ == "__main__":
    tol = 1e-6
    cases = {
        "notebook-8tip": "(((A:1.,B:1.):1.,(C:1.,D:1.):1.):1., ((E:1.,F:1.):1., (G:1.,H:1.):1.):1.):0.01;",
        "unbalanced": "(A:0.5,(B:1.0,(C:2.0,(D:0.3,E:1.5):0.7):0.4):0.9):0.0;",
        "multifurcation": "((A:1.0,B:1.0,C:1.0):0.5,(D:2.0,E:1.0):1.0):0.0;",
    }

    worst = 0.0
    for name, nwk in cases.items():
        worst = max(worst, check_tree(name, nwk))
    for name, nwk in cases.items():
        worst = max(worst, check_tree_ou(name, nwk))

    worst = max(worst, check_ou_regimes("notebook-8tip", cases["notebook-8tip"],
                                        shift_leaves=["E", "F", "G", "H"]))
    worst = max(worst, check_ou_regimes("unbalanced", cases["unbalanced"],
                                        shift_leaves=["C", "D", "E"]))
    worst = max(worst, timing_demo())

    # BM-limit is an O(alpha) consistency check, not an exact-equality oracle.
    limit_tol = 1e-4
    limit_err = check_bm_limit(cases["notebook-8tip"])

    print()
    ok = worst < tol and limit_err < limit_tol
    if ok:
        print(f"PASS: pruning matches dense oracle (worst {worst:.3e} < {tol:g}); "
              f"BM-limit {limit_err:.3e} < {limit_tol:g}")
    else:
        print(f"FAIL: worst oracle error {worst:.3e} (tol {tol:g}), "
              f"BM-limit {limit_err:.3e} (tol {limit_tol:g})")
        raise SystemExit(1)
