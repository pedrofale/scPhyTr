"""End-to-end: does ``tl.detect_modes`` find a planted event coming in through an AnnData?

The unit tests in ``test_modes_model.py`` exercise the sampler on its own array tree. This one goes
through the whole public path -- ete3 tree, AnnData, size factors, the cells-to-leaves reduction --
because that is where the wiring can be wrong while every component is right.

Note it matches the answer by **clade**, not by branch index: ``detect_modes`` renumbers nodes when
it converts the ete3 tree, so an index from one tree means nothing in the other.
"""
import os
import tempfile

import numpy as np
import pandas as pd

from scphytr.modes import Tree as ArrayTree, simulate_tips, candidate_branches


def _newick(t, v=0):
    if len(t.children[v]) == 0:
        return f"{t.name[v]}:{t.dist[v]:.10f}"
    inner = ",".join(_newick(t, c) for c in t.children[v])
    return f"({inner})" + (f":{t.dist[v]:.10f}" if t.parent[v] >= 0 else ";")


def test_detect_modes_finds_a_planted_event():
    import anndata
    import scphytr as ph

    n_tips, n_genes, frac, shift = 128, 200, 0.25, 1.2
    at = ArrayTree.balanced(n_tips, 1.0).scale_height(1.0)
    sets = at.leaf_sets()
    b = int(next(v for v in candidate_branches(at, 8) if len(sets[v]) == n_tips // 4))
    true_clade = frozenset(at.name[at.leaves[i]] for i in sets[b])

    rng = np.random.default_rng(0)
    delta = np.zeros((at.n_nodes, n_genes))
    resp = rng.choice(n_genes, int(frac * n_genes), replace=False)
    delta[b, resp] = rng.normal(0.0, shift, size=len(resp))
    lat = simulate_tips(at, 2.0, 0.75, rng.normal(size=n_genes), delta, rng=rng, root="stationary")
    E = np.exp(lat - lat.mean(0))
    counts = rng.poisson(60 * E / E.mean(0)).astype(float)

    fp = tempfile.NamedTemporaryFile("w", suffix=".nwk", delete=False)
    fp.write(_newick(at)); fp.close()
    try:
        tree = ph.Tree(fp.name)
    finally:
        os.unlink(fp.name)

    names = [at.name[l] for l in at.leaves]
    ad = anndata.AnnData(X=counts,
                         obs=pd.DataFrame({"species": names}, index=names),
                         var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]))
    ph.pp.setup_anndata(ad, tree)
    ph.tl.detect_modes(ad, evidence_only=True)

    m = ad.uns["modes"]
    order = np.argsort(-m["evidence"])
    ranked = [frozenset(m["clades"][i]) for i in order]
    assert ranked.index(true_clade) == 0            # the true clade tops the branch scan
    assert len(m["clades"]) == len(m["branches"])
    assert set(m["leaf_names"]) == set(names)
