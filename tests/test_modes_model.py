"""Model 1: the linear-mean identity, the clade degeneracy it implies, and event recovery."""
import numpy as np

from scphytr.modes._tree import Tree
from scphytr.modes.design import shift_design, candidate_branches
from scphytr.modes._ou import node_optima, tip_mean
from scphytr.modes.simulate import simulate_tips
from scphytr.modes.model import fit_model1, branch_evidence


def test_design_is_the_mean():
    """theta0 + U(alpha) delta must equal the OU tip mean exactly, for any alpha and any shifts."""
    tree = Tree.balanced(32, 1.0).scale_height(1.0)
    rng = np.random.default_rng(0)
    for alpha in (0.2, 1.0, 5.0):
        U = shift_design(tree, alpha)
        d = np.zeros((tree.n_nodes, 1))
        d[rng.choice(np.arange(1, tree.n_nodes), 5, replace=False), 0] = rng.normal(size=5)
        ref = tip_mean(tree, alpha, node_optima(tree, np.array([0.7]), d))[:, 0]
        assert np.allclose(ref, 0.7 + U @ d[:, 0], atol=1e-12)


def test_parent_column_lies_in_the_span_of_its_children():
    """On an ultrametric tree U[:, b] is a scaled clade indicator, so a parent event is exactly
    mimicked by events on any partition of its clade. This is why the sampler needs block moves."""
    tree = Tree.balanced(64, 1.0).scale_height(1.0)
    U = shift_design(tree, 2.0)
    p = next(v for v in range(1, tree.n_nodes) if len(tree.children[v]) == 2)
    A = U[:, list(tree.children[p])]
    resid = U[:, p] - A @ np.linalg.lstsq(A, U[:, p], rcond=None)[0]
    assert np.linalg.norm(resid) / np.linalg.norm(U[:, p]) < 1e-10


def _planted(shift=0.6, n_tips=128, G=200, frac=0.25, alpha=3.0, sigma=0.75, seed=0):
    tree = Tree.balanced(n_tips, 1.0).scale_height(1.0)
    br = candidate_branches(tree, 8)
    sets = tree.leaf_sets()
    b = int(next(v for v in br if len(sets[v]) == n_tips // 4))
    rng = np.random.default_rng(seed)
    delta = np.zeros((tree.n_nodes, G))
    resp = rng.choice(G, int(frac * G), replace=False)
    delta[b, resp] = rng.normal(0.0, shift, size=len(resp))
    Y = simulate_tips(tree, alpha, sigma, rng.normal(size=G), delta, rng=rng, root="stationary")
    return tree, br, b, Y, resp


def test_recovers_a_planted_event():
    tree, br, b, Y, resp = _planted()
    res = fit_model1(Y, tree, branches=br, alpha_grid=(3.0,), tau_grid=(0.02,),
                     n_iter=400, burn=150, seed=1)
    i = list(br).index(b)
    assert res.p_z[i] > 0.9                                  # the right branch...
    assert abs(res.n_event_draws.mean() - 1.0) < 0.3         # ...and only the right branch
    is_resp = np.zeros(Y.shape[1], bool); is_resp[resp] = True
    pg = res.p_gamma[i]
    assert pg[is_resp].mean() > 3 * pg[~is_resp].mean()      # gamma separates responders


def test_branch_evidence_localises_without_mcmc():
    tree, br, b, Y, _ = _planted(shift=0.3)
    bs, ev = branch_evidence(Y, tree, branches=br, alpha_grid=(1.0, 3.0), tau_grid=(0.02, 0.2),
                             rho=0.1)
    assert int(bs[int(np.argmax(ev))]) == b
