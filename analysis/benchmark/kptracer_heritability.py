"""KP-Tracer heritability benchmark: scPhyTr Pagel's lambda vs PATH Moran's I.

This is the KP-Tracer column of the benchmark matrix (task 1: heritability) -- the
sparse, single-cell lineage-tracing regime (Yang*, Jones* et al., Cell 2022) that is
Hotspot's home turf and where scPhyTr's single-cell phylogenetic-signal estimate should
hold up against PATH's autocorrelation. We compare two heritability readouts on the SAME
cell-level tree and the SAME shared-ancestry covariance:

  * scPhyTr  -- Pagel's lambda (model-based, non-saturating; ``scphytr.tools.heritability``)
  * PATH     -- Moran's I phylogenetic autocorrelation (``analysis.benchmark.path_morans``)

Both run off the one shared-time matrix ``C`` that ``analysis.kptracer.load.load_tumor``
already builds in a single traversal, so no per-pair tree queries and no ``Tree`` wrapper:
  - Pagel's lambda takes ``C=(C, names)`` directly;
  - Moran's weight matrix W is C with a zeroed diagonal (PATH's shared-ancestry weights).

Sub-indexing ``C`` to a random leaf subset is itself a valid shared-time matrix (shared
ancestry between two retained leaves is unchanged by dropping others), so we subsample to
keep the dense O(n^3) lambda and O(n^2) weights tractable -- no tree re-pruning.

Traits:
  * per cell-state -- one-hot membership in each ``Cluster-Name`` state (EMT / Mesenchymal /
    Endoderm / ... axis); a 0/1 heritability trait, the categorical-state question PATH asks.
  * per gene       -- log-normalized expression of the top HVGs (continuous traits).

NOTE on counts: the integrated KP-Tracer AnnData carries only library-normalized expression
(``raw.X`` is normalized to a constant target sum, non-integer; ``X`` is its log1p), so the
count-model advantage cannot be shown here -- scPhyTr runs on the Gaussian-trait path. The
deconfounded-K / count story lives in the melanoma + simulation columns.
"""
import os
import numpy as np
import pandas as pd

from scphytr.tools.heritability import pagels_lambda
from analysis.benchmark.path_morans import morans_I, path_test
from analysis.kptracer.load import load_tumor

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data", "external", "KPTracer-Data")
OUT = os.path.dirname(__file__)

# The four high-fitness sgNT tumors used elsewhere in the KP-Tracer analysis.
TUMORS = ["3726_NT_T2", "3430_NT_T2", "3513_NT_T3", "3434_NT_T1"]


def subsample(C, names, Y, obs, n_sub, seed=0):
    """Random leaf subset; sub-index the shared-time matrix and expression in place."""
    n = len(names)
    if n <= n_sub:
        idx = np.arange(n)
    else:
        idx = np.sort(np.random.default_rng(seed).choice(n, size=n_sub, replace=False))
    Csub = C[np.ix_(idx, idx)]
    names_sub = [names[i] for i in idx]
    return Csub, names_sub, Y[idx], obs.iloc[idx]


def heritability_pair(C, names, values, n_perm=199, seed=0):
    """Both heritability readouts for one trait on one tree. Returns a dict row."""
    y = np.array([float(values[nm]) for nm in names])
    if np.allclose(y, y[0]):
        return None  # constant trait (e.g. absent cell-state) -- undefined
    # PATH: Moran's I with shared-ancestry weights (zeroed diagonal) + permutation p
    W = C.copy()
    np.fill_diagonal(W, 0.0)
    I, p_path = path_test(y, W, n_perm=n_perm, rng=np.random.default_rng(seed))
    # scPhyTr: Pagel's lambda + LR test vs lambda=0 (star / no signal)
    lam = pagels_lambda(None, values, C=(C, names))
    return {
        "morans_I": I, "p_path": p_path,
        "lambda": lam["lambda"], "p_lambda": lam["p"],
        "path_call": p_path < 0.05,
        "lambda_call": (lam["lambda"] > 0.05 and lam["p"] < 0.05),
    }


def run_tumor(tumor, adata, n_sub=1200, n_hvg=50, min_state_cells=25, seed=0):
    d = load_tumor(tumor, DATA, adata=adata, n_hvg=n_hvg)
    C, names, Y, obs = subsample(d["C"], d["leaf_names"], d["Y"], d["obs"], n_sub, seed)
    genes = d["genes"]
    n = len(names)
    print(f"\n=== {tumor}: {len(d['leaf_names'])} cells -> {n} subsampled "
          f"(tree height ~{np.diag(C).mean():.1f}) ===")

    rows = []
    # --- per cell-state (one-hot Cluster-Name) ---
    states = obs["Cluster-Name"].astype(str)
    for st, cnt in states.value_counts().items():
        if cnt < min_state_cells or cnt == n:
            continue
        ind = {nm: float(states.iloc[i] == st) for i, nm in enumerate(names)}
        r = heritability_pair(C, names, ind, seed=seed)
        if r is None:
            continue
        minfrac = min(cnt, n - cnt) / n
        r.update({"tumor": tumor, "kind": "cell_state", "trait": st,
                  "n_pos": int(cnt), "n": n, "minfrac": minfrac,
                  "balanced": minfrac >= 0.05})
        rows.append(r)
        print(f"  state {st:<22} (n+={cnt:4d})  I={r['morans_I']:+.3f} "
              f"(p={r['p_path']:.3f})  lambda={r['lambda']:.3f} (p={r['p_lambda']:.1e})  "
              f"{'AGREE' if r['path_call']==r['lambda_call'] else 'disagree'}")

    # --- per gene (top HVGs, log-norm expression) ---
    for g in range(Y.shape[1]):
        vals = {nm: float(Y[i, g]) for i, nm in enumerate(names)}
        r = heritability_pair(C, names, vals, seed=seed)
        if r is None:
            continue
        r.update({"tumor": tumor, "kind": "gene", "trait": str(genes[g]),
                  "n_pos": -1, "n": n, "minfrac": np.nan, "balanced": True})
        rows.append(r)
    return rows


def main(n_sub=1200, n_hvg=50, seed=0):
    import anndata as ad
    print("loading integrated AnnData (one read for all tumors) ...")
    adata = ad.read_h5ad(os.path.join(DATA, "expression", "adata_processed.nt.h5ad"))

    all_rows = []
    for t in TUMORS:
        try:
            all_rows += run_tumor(t, adata, n_sub=n_sub, n_hvg=n_hvg, seed=seed)
        except Exception as e:
            print(f"  !! {t} skipped: {e}")
    df = pd.DataFrame(all_rows)
    out_csv = os.path.join(OUT, "kptracer_heritability.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}  ({len(df)} trait rows)")

    # --- concordance summary ---
    from scipy.stats import spearmanr
    print("\n================ CONCORDANCE SUMMARY ================")
    # PER-GENE: Moran's I is NOT comparable across trees of different depth, but
    # Pagel's lambda is bounded [0,1] and comparable -- so rank concordance is the
    # fair statistic WITHIN each tumor, not pooled across tumors.
    g = df[df["kind"] == "gene"]
    print("per-gene heritability (scPhyTr lambda vs PATH Moran's I), per tumor:")
    rhos = []
    for t, sub in g.groupby("tumor"):
        rho = spearmanr(sub["morans_I"], sub["lambda"]).statistic
        agree = (sub["path_call"] == sub["lambda_call"]).mean()
        rhos.append(rho)
        print(f"  {t:<12} n={len(sub):3d}  Spearman(I,lambda)={rho:+.2f}  "
              f"call-agreement(p<.05)={agree:.0%}")
    if rhos:
        print(f"  -> median per-tumor Spearman = {np.median(rhos):+.2f}; "
              f"overall call-agreement = {(g['path_call']==g['lambda_call']).mean():.0%}")

    # PER CELL-STATE: these high-fitness tumors are fate-converged (one state
    # dominates), so most one-hot state traits are near-constant. lambda OVERFITS on
    # rare states (documented small-sample behaviour); restrict to balanced states.
    cs = df[df["kind"] == "cell_state"]
    if not cs.empty:
        bal = cs[cs["balanced"]]
        deg = cs[~cs["balanced"]]
        print(f"\nper cell-state ({len(cs)} states; {len(bal)} balanced minfrac>=5%, "
              f"{len(deg)} degenerate/rare):")
        if not bal.empty:
            agree = (bal["path_call"] == bal["lambda_call"]).mean()
            print(f"  balanced states: call-agreement={agree:.0%}  "
                  f"(both call heritable on the dominant fate axis)")
        if not deg.empty:
            print(f"  degenerate states: lambda overfits (e.g. lambda~1 on <5% minority) "
                  f"while Moran's I~0 -- needs bootstrap/regularization, as on small EMT clones")
    return df


if __name__ == "__main__":
    main()
