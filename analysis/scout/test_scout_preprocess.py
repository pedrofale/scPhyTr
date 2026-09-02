"""Our preprocessing port must match SCOUT's R implementation exactly.

Fixtures in ``tests/fixtures/`` were produced by running SCOUT's own ``lineage_smooth`` (extracted
verbatim from SCOUT/R/SCOUT_EM_utils.R) under R 4.5 with ape; see ``generate_fixtures.R``. This test
therefore pins our Python port against the reference implementation without needing R installed.
"""
import os

import numpy as np
import pandas as pd

from sparseou.tree import Tree
from sparseou.preprocess import lineage_smooth, patristic, log_normalize

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def _load():
    tree = Tree.from_newick(open(os.path.join(FIX, "smooth_tree.nwk")).read())
    tips = [tree.name[l] for l in tree.leaves]
    X = pd.read_csv(os.path.join(FIX, "smooth_X.csv"), index_col=0).loc[tips].values
    return tree, tips, X


def test_lineage_smooth_matches_scout_R():
    tree, tips, X = _load()
    for k in (2, 3, 5):
        ref = pd.read_csv(os.path.join(FIX, f"smooth_R_k{k}.csv"), index_col=0).loc[tips].values
        got = lineage_smooth(tree, X, k)
        err = np.abs(ref - got).max()
        assert err < 1e-8, f"k={k}: max abs diff {err:.3e} vs SCOUT's R"


def test_patristic_is_symmetric_and_additive():
    tree = Tree.from_newick("((a:1,b:1):2,(c:0.5,d:0.5):2.5);")
    D = patristic(tree)
    assert np.allclose(D, D.T)
    assert np.allclose(np.diag(D), 0)
    tips = [tree.name[l] for l in tree.leaves]
    i = {t: k for k, t in enumerate(tips)}
    assert abs(D[i["a"], i["b"]] - 2.0) < 1e-12
    assert abs(D[i["c"], i["d"]] - 1.0) < 1e-12
    assert abs(D[i["a"], i["c"]] - 6.0) < 1e-12


def test_smoothing_preserves_shape_and_is_finite():
    tree, tips, X = _load()
    S = lineage_smooth(tree, X, 3)
    assert S.shape == X.shape and np.isfinite(S).all()


def test_log_normalize_is_log1p():
    counts = np.array([[0.0, 1.0], [5.0, 100.0]])
    assert np.allclose(log_normalize(counts), np.log1p(counts))
