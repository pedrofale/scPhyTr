"""The regime painting and the tree read-outs built on it.

The painting is the part worth testing: everything else here is drawing. A branch above a regime
split must not be attributed to either side, because that is exactly the confusion the painting
exists to make visible.
"""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import LineCollection

import scphytr.plotting as pl
from scphytr.modes._tree import Tree
from scphytr.modes.baseline import paint_regimes

# ((A,B),(C,D)) -- two clades of two, so the root and nothing else spans both regimes
NWK = "((A:1.0,B:1.0)AB:1.0,(C:1.0,D:1.0)CD:1.0)root:0.0;"


def _tree():
    return Tree.from_newick(NWK)


def test_to_newick_round_trips():
    t = _tree()
    t2 = Tree.from_newick(t.to_newick())
    assert t2.n_nodes == t.n_nodes and t2.n_leaves == t.n_leaves
    assert [t2.name[l] for l in t2.leaves] == [t.name[l] for l in t.leaves]
    assert np.allclose(sorted(t2.depth), sorted(t.depth))


def test_paint_regimes_mixed_none_marks_the_split():
    t = _tree()
    lr = ["x", "x", "y", "y"]
    root_painted, uniq = paint_regimes(t, lr)                 # default: mixed -> root regime
    none_painted, _ = paint_regimes(t, lr, mixed=None)
    root = int(np.flatnonzero(t.parent < 0)[0])
    assert root_painted[root] == uniq.index("x")              # the old behaviour is unchanged
    assert none_painted[root] == -1                           # the new one marks it instead
    # every node below the root sits in exactly one regime, so the two agree there
    assert np.array_equal(np.delete(root_painted, root), np.delete(none_painted, root))


def test_paint_regimes_rejects_bad_mixed():
    with pytest.raises(ValueError):
        paint_regimes(_tree(), ["x", "x", "y", "y"], mixed="nearest")


def test_paint_from_leaves_agrees_with_paint_regimes():
    t = _tree()
    lr = ["x", "x", "y", "y"]
    codes, uniq = paint_regimes(t, lr, mixed=None)
    by_name = {t.name[v]: (uniq[c] if c >= 0 else None) for v, c in enumerate(codes)}
    painted = pl.paint_from_leaves(t, dict(zip(["A", "B", "C", "D"], lr)))
    assert {nd.name: lab for nd, lab in painted.items()} == by_name
    assert by_name["root"] is None and by_name["AB"] == "x" and by_name["CD"] == "y"


def test_paint_from_leaves_accepts_a_sequence_and_checks_its_length():
    t = _tree()
    painted = pl.paint_from_leaves(t, ["x", "x", "y", "y"])
    assert sorted(v for v in painted.values() if v) == ["x", "x", "x", "y", "y", "y"]
    with pytest.raises(ValueError):
        pl.paint_from_leaves(t, ["x", "y"])


def test_regime_palette_is_stable_and_shareable():
    a = pl.regime_palette(["x", "y"])
    b = pl.regime_palette(["x", "y", "z"])
    assert a["x"] == b["x"] and a["y"] == b["y"]              # a shared palette keeps colours put
    assert len(pl.regime_palette(["x", "x", "y"])) == 2       # first-seen order, de-duplicated


def test_plot_tree_colours_branches_by_regime():
    t = _tree()
    ax = pl.plot_tree(t, regimes=["x", "x", "y", "y"], label_leaves=False, tip_strip=True)
    cols = [c for c in ax.collections if isinstance(c, LineCollection)]
    drawn = np.vstack([c.get_colors() for c in cols])
    pal = pl.regime_palette(["x", "y"])
    for lab in ("x", "y"):
        assert np.any(np.all(np.isclose(drawn, pal[lab]), axis=1)), f"{lab} not drawn"
    grey = np.all(np.isclose(drawn, (0.75, 0.75, 0.75, 1.0)), axis=1)
    assert grey.sum() >= 1, "the branch spanning both regimes should be drawn grey"
    assert ax.get_legend() is not None


def test_plot_tree_rejects_regimes_with_continuous_values():
    t = _tree()
    with pytest.raises(ValueError):
        pl.plot_tree(t, regimes=["x", "x", "y", "y"], node_values={})


def test_expression_tree_draws_one_strip_per_key():
    cas = pytest.importorskip("cassiopeia")
    t = _tree()
    meta = pd.DataFrame({"regime": ["x", "x", "y", "y"], "g1": [0.0, 1.0, 2.0, 3.0]},
                        index=["A", "B", "C", "D"])
    ax = pl.expression_tree(t, ["regime", "g1"], cell_meta=meta, figsize=(4, 4))
    labels = [tx.get_text().strip() for tx in ax.texts]
    assert "regime" in labels and "g1" in labels


def test_expression_tree_reports_missing_values():
    pytest.importorskip("cassiopeia")
    t = _tree()
    meta = pd.DataFrame({"g1": [0.0, 1.0]}, index=["A", "B"])
    with pytest.raises(KeyError):
        pl.expression_tree(t, ["g1"], cell_meta=meta)
    with pytest.raises(KeyError):
        pl.expression_tree(t, ["absent"], cell_meta=pd.DataFrame(index=["A", "B", "C", "D"]))


def _adata_on(tree):
    """Two cells per tip, the shape `pp.setup_anndata` leaves behind."""
    import anndata as ad
    obs = pd.DataFrame({"species": ["A", "A", "B", "B", "C", "C", "D", "D"],
                        "cell_type": ["n", "n", "n", "g", "g", "g", "g", "g"]})
    a = ad.AnnData(np.arange(24, dtype=float).reshape(8, 3), obs=obs,
                   var=pd.DataFrame(index=["g1", "g2", "g3"]))
    a.uns["tree"] = tree
    return a


def test_adata_leaf_values_handles_categorical_obs():
    from scphytr.plotting.tree import _adata_leaf_values
    a = _adata_on(_tree())
    assert _adata_leaf_values(a, "cell_type") == {"A": "n", "B": "n", "C": "g", "D": "g"}
    assert set(_adata_leaf_values(a, "g1")) == {"A", "B", "C", "D"}      # genes still numeric


def test_expression_tree_from_adata():
    pytest.importorskip("cassiopeia")
    ax = pl.expression_tree(_adata_on(_tree()), ["g1", "cell_type"], figsize=(4, 4))
    assert {"g1", "cell_type"} <= {tx.get_text().strip() for tx in ax.texts}


def test_plot_tree_regime_cmap_selects_the_palette():
    t = _tree()
    ax = pl.plot_tree(t, regimes=["x", "x", "y", "y"], regime_cmap="Set2", label_leaves=False)
    drawn = np.vstack([c.get_colors() for c in ax.collections
                       if isinstance(c, LineCollection)])
    want = pl.regime_palette(["x", "y"], cmap="Set2")
    assert np.any(np.all(np.isclose(drawn, want["x"]), axis=1))
