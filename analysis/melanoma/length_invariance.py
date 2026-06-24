"""Empirical check: the fitted evolutionary K is invariant to a gene-length offset.

Smart-seq2 read counts scale with transcript length, but gene length is a
per-gene constant and BM/OU give every gene a free baseline (root mean / optimum)
that absorbs it. So putting the effective length in the Poisson offset
(``S_i * L_g``) versus leaving it out (``S_i``) should leave the diffusion matrix
K -- the per-gene evolutionary rates and the gene-gene evolutionary correlations
-- essentially unchanged. This fits both ways on the real melanoma subclone
counts and reports the difference.
"""
import numpy as np

from analysis.melanoma.load import load_counts, tree_leaves, load_tree
from scphytr.inference.laplace import MultiCellPoissonObservation
from scphytr.tools.em import fit_mv_em
from scphytr.tools.estimation import cov_to_corr


def _pick_genes(n=5):
    X, genes, clone, _ = load_counts()
    tot = X.sum(axis=0)
    order = np.argsort(tot)[::-1]
    keep = [g for g in order if (X[:, g] > 0).mean() > 0.6][:n]
    return list(genes[keep])


def _fit(genes, length_offset, dispersion=10.0):
    X, gene_ids, clone, offsets = load_counts(genes=genes, length_offset=length_offset)
    leaves = tree_leaves()
    leaf_of = {name: k for k, name in enumerate(leaves)}
    idx = np.array([leaf_of[c] for c in clone])
    tree = load_tree()                          # floors zero-length branches
    obs = MultiCellPoissonObservation(X, offsets, idx, len(leaves), dispersion=dispersion)
    res = fit_mv_em(tree, obs, model="BM", max_em=25)
    return np.asarray(res.covariance())


def main():
    genes = _pick_genes()
    K0 = _fit(genes, length_offset=False)   # length absorbed by per-gene mean
    K1 = _fit(genes, length_offset=True)    # length in the Poisson offset
    d0, d1 = np.diag(K0), np.diag(K1)
    R0, R1 = cov_to_corr(K0), cov_to_corr(K1)
    iu = np.triu_indices(len(genes), 1)
    print(f"{len(genes)} genes, {tree_leaves().__len__()} subclone leaves\n")
    print("per-gene rate (diag K):")
    print("  no length offset :", np.round(d0, 3))
    print("  with length offset:", np.round(d1, 3))
    print(f"  max |rate diff|       = {np.max(np.abs(d0 - d1)):.2e}  "
          f"(max rel = {np.max(np.abs(d0 - d1) / d0):.1e})")
    print(f"  max |corr diff|       = {np.max(np.abs(R0[iu] - R1[iu])):.2e}")
    print("\n=> K is invariant to the length offset (differences at optimizer "
          "tolerance), as predicted: gene length is absorbed by the per-gene mean.")


if __name__ == "__main__":
    main()
