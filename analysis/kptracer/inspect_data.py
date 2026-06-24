"""Quick structural inspection of the KP-Tracer NT AnnData + a tree."""
import os
import numpy as np
import anndata
import ete3

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data", "external", "KPTracer-Data")

ad = anndata.read_h5ad(os.path.join(DATA, "expression", "adata_processed.nt.h5ad"), backed="r")
print("shape (cells, genes):", ad.shape)
print("obs columns:", list(ad.obs.columns))
print("var columns:", list(ad.var.columns))
print("obsm:", list(ad.obsm.keys()))
print("layers:", list(ad.layers.keys()))
print("raw:", None if ad.raw is None else ad.raw.shape)
X = ad.X[:200]
X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
print(f"X[:200] min={X.min():.3f} max={X.max():.3f} mean={X.mean():.3f} "
      f"integer-valued={np.allclose(X, np.round(X))} frac_zero={(X==0).mean():.2f}")
if "Tumor" in ad.obs:
    vc = ad.obs["Tumor"].value_counts()
    nt = vc[[t for t in vc.index if "_NT_" in str(t)]]
    print("\ntop NT tumors by #cells:")
    print(nt.head(15))
if "leiden_sub" in ad.obs:
    print("\n#leiden_sub clusters:", ad.obs["leiden_sub"].nunique())

# a representative tree
for tumor in ["3726_NT_T2", "3513_NT_T3", "3430_NT_T2"]:
    fp = os.path.join(DATA, "trees", f"{tumor}_tree.nwk")
    if not os.path.exists(fp):
        continue
    t = ete3.Tree(fp, format=1)
    leaves = t.get_leaves()
    dists = [float(n.dist) for n in t.traverse() if not n.is_root()]
    depths = []
    for n in t.traverse("preorder"):
        n.add_feature("d", float(n.dist) if n.is_root() else n.up.d + float(n.dist))
    depths = [l.d for l in leaves]
    print(f"\n[{tumor}] #leaves={len(leaves)} "
          f"branch len: min={min(dists):.3g} max={max(dists):.3g} "
          f"frac_zero={(np.array(dists)==0).mean():.2f} | "
          f"root-to-tip depth: min={min(depths):.3g} max={max(depths):.3g} "
          f"ultrametric={np.allclose(depths, depths[0])}")
    ex = leaves[0].name
    print(f"   example leaf name: {ex}  in_adata={ex in set(ad.obs_names)}")
