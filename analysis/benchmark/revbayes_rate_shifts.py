"""scPhyTr vs RevBayes: clade-specific rate-shift detection, head-to-head.

Both methods fit O'Meara's multi-rate Brownian motion -- the diffusion rate sigma^2
changes at the base of a clade -- but with opposite inference philosophies:

  * RevBayes  -- Bayesian reversible-jump MCMC over the number AND location of rate
    shifts: each branch carries an RJ mixture rate multiplier (1 = no shift, else a
    lognormal draw), and the posterior mean of the shift indicator is a per-branch
    shift probability. Full uncertainty, but an MCMC per dataset.
  * scPhyTr   -- maximum-penalized-likelihood: greedy forward selection of shifts
    scored by BIC, each likelihood the linear-time Felsenstein pruning
    (``scphytr.tools.model_selection.detect_rate_shifts``). One point configuration,
    no MCMC -- milliseconds.

We simulate a continuous trait on a random tree with ONE known rate shift (a clade
at ``rate_clade`` against a ``rate_bg`` background), hand the *same* tree and trait
to both tools, and score whether each localizes the shift to the true clade, plus
runtime. A real-data sanity check (a melanoma-subline gene) runs both on an
empirical tree.

Requires the RevBayes binary; set ``$RB`` (default: the unpacked mac-arm64 build).
"""
import os
import re
import json
import time
import shutil
import subprocess
import tempfile

import numpy as np
from ete3 import Tree as ETree

from scphytr.utils.tree import Tree
from scphytr.utils.pruning import paint_regimes
from scphytr.tools.model_selection import fit_bm, detect_rate_shifts

RB = os.environ.get("RB") or shutil.which("rb") or os.path.expanduser(
    "~/opt/revbayes-v1.4.0/bin/rb")
OUT = os.path.dirname(__file__)


# --------------------------------------------------------------------------- #
# Simulation: one known rate shift on a random tree
# --------------------------------------------------------------------------- #

def simulate(n_tips=40, rate_bg=0.4, rate_clade=9.0, seed=0):
    rng = np.random.default_rng(seed)
    et = ETree(); et.populate(n_tips, random_branches=True)
    for nd in et.traverse():
        if not nd.is_root():
            nd.dist = round(max(float(nd.dist), 0.05), 4)
    for i, l in enumerate(et.get_leaves()):
        l.name = f"t{i}"
    tw = Tree(); tw.phylotree = et; tw.root = et.get_tree_root()
    internals = [nd for nd in et.traverse() if not nd.is_leaf() and not nd.is_root()]
    # a sizeable, balanced clade for the shift
    shift = max(internals, key=lambda c: min(len(c.get_leaves()), n_tips - len(c.get_leaves())))
    regimes, nreg = paint_regimes(tw, [shift])
    rates = {0: rate_bg, 1: rate_clade}
    values = {}

    def desc(node, pv):
        if node is tw.root:
            v = 0.0
        else:
            r = rates[regimes[node]]
            v = pv + rng.normal(0.0, np.sqrt(r * node.dist))
        if node.is_leaf():
            values[node.name] = v
        for c in node.children:
            desc(c, v)
    desc(tw.root, 0.0)
    return tw, values, frozenset(shift.get_leaf_names())


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 0.0


# --------------------------------------------------------------------------- #
# scPhyTr
# --------------------------------------------------------------------------- #

def run_scphytr(tree, values, true_clade):
    t0 = time.time()
    res = detect_rate_shifts(tree, values, max_shifts=4, criterion="bic")
    dt = time.time() - t0
    shifts = res["shifts"]
    best_jac, best_clade = 0.0, frozenset()
    for nd in shifts:
        j = jaccard(nd.get_leaf_names(), true_clade)
        if j > best_jac:
            best_jac, best_clade = j, frozenset(nd.get_leaf_names())
    return {"method": "scphytr", "n_shifts": len(shifts), "jaccard": best_jac,
            "rates": res["fit"].params.get("rates"), "seconds": dt,
            "selected_bm_over_shift": len(shifts) == 0}


# --------------------------------------------------------------------------- #
# RevBayes
# --------------------------------------------------------------------------- #

# RevBayes random local clock: a reversible-jump rate multiplier per branch, but
# rates are INHERITED (node_rate[i] = node_rate[parent] * mult[i]), so a single jump
# at a clade's base raises the rate of the whole clade -- the rate-shift model. The
# tree's node indices satisfy parent(i) > i, so a descending pass defines each
# parent's rate before its children.
_REV_TEMPLATE = """\
trait <- readContinuousCharacterData("{nex}")
tree  <- readTrees("{nwk}")[1]
nn = tree.nnodes()
nb = nn - 1
root = tree.getRootIndex()
moves = VectorMoves()
log_base ~ dnUniform(-10, 5)
moves.append( mvSlide(log_base, weight=3) )
base := exp(log_base)
H <- 1.25
p_no <- 1.0 - 2.0 / nb      # parsimonious prior: ~2 expected rate shifts
node_rate[root] := base
for (i in (nn-1):1) {{
    mult[i] ~ dnReversibleJumpMixture(1.0, dnLognormal(0.0, H), p_no)
    moves.append( mvRJSwitch(mult[i], weight=1) )
    moves.append( mvScale(mult[i], lambda=0.5, weight=1) )
    is_shift[i] := ifelse(mult[i] == 1.0, 0.0, 1.0)
    node_rate[i] := node_rate[tree.parent(i)] * mult[i]
}}
for (i in 1:nb) {{ branch_rate[i] := node_rate[i] }}
seq ~ dnPhyloBrownianREML(tree, branchRates=branch_rate, nSites=1)
seq.clamp(trait)
mymodel = model(base)
monitors = VectorMonitors()
monitors.append( mnModel(filename="{log}", printgen=20) )
mc = mcmc(mymodel, monitors, moves)
mc.burnin(generations={burnin}, tuningInterval=200)
mc.run(generations={ngen})
write("idx", "taxa", filename="{clades}", separator="\\t")
write("\\n", filename="{clades}", append=TRUE)
for (i in 1:nb) {{
    write(i, tree.getDescendantTaxa(i), filename="{clades}", separator="\\t", append=TRUE)
    write("\\n", filename="{clades}", append=TRUE)
}}
q()
"""


def _write_inputs(tree, values, wd):
    nwk = os.path.join(wd, "tree.nwk")
    tree.phylotree.write(format=5, outfile=nwk)
    nex = os.path.join(wd, "trait.nex")
    with open(nex, "w") as f:
        f.write("#NEXUS\nBEGIN DATA;\n")
        f.write(f"DIMENSIONS NTAX={len(values)} NCHAR=1;\n")
        f.write("FORMAT DATATYPE=CONTINUOUS;\nMATRIX\n")
        for k, v in values.items():
            f.write(f"{k} {v:.6f}\n")
        f.write(";\nEND;\n")
    return nwk, nex


def _parse_clades(path):
    idx2clade = {}
    with open(path) as f:
        next(f, None)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            taxa = re.findall(r"[\w.]+", parts[1].strip())
            idx2clade[int(parts[0])] = frozenset(taxa)
    return idx2clade


def run_revbayes(tree, values, true_clade, ngen=20000, burnin=3000, rb=RB, keep_dir=None):
    if not os.path.exists(rb):
        raise FileNotFoundError(f"RevBayes binary not found at {rb}; set $RB.")
    wd = keep_dir or tempfile.mkdtemp(prefix="rb_rate_")
    os.makedirs(wd, exist_ok=True)
    nwk, nex = _write_inputs(tree, values, wd)
    logf = os.path.join(wd, "rj.log"); cladesf = os.path.join(wd, "clades.txt")
    rev = os.path.join(wd, "rateshift.Rev")
    with open(rev, "w") as f:
        f.write(_REV_TEMPLATE.format(nex=nex, nwk=nwk, log=logf, clades=cladesf,
                                     burnin=burnin, ngen=ngen))
    t0 = time.time()
    subprocess.run([rb, rev], check=True, capture_output=True, text=True)
    dt = time.time() - t0

    # posterior per-branch shift probability
    import csv
    with open(logf) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    cols = [c for c in rows[0] if c.startswith("is_shift")]
    # column name is_shift[i]; map to branch index
    def col_idx(c):
        return int(re.search(r"\[(\d+)\]", c).group(1))
    post = {col_idx(c): np.mean([float(r[c]) for r in rows]) for c in cols}
    rate_cols = [c for c in rows[0] if c.startswith("branch_rate")]
    post_rate = {col_idx(c): np.mean([float(r[c]) for r in rows]) for c in rate_cols}
    idx2clade = _parse_clades(cladesf)
    # branch with highest posterior shift probability, and the prob on the true clade
    best_idx = max(post, key=post.get)
    map_clade = idx2clade.get(best_idx, frozenset())
    true_prob = max((post[i] for i, cl in idx2clade.items() if cl == true_clade), default=0.0)
    # robust "did it infer the rate elevation": posterior mean branch rate, clade vs background
    in_clade = [i for i, cl in idx2clade.items() if cl and cl <= true_clade]
    bg = [i for i, cl in idx2clade.items() if cl and not (cl <= true_clade)]
    rate_ratio = (np.mean([post_rate[i] for i in in_clade]) /
                  max(np.mean([post_rate[i] for i in bg]), 1e-9)) if in_clade and bg else np.nan
    return {"method": "revbayes", "map_post": post[best_idx],
            "map_jaccard": jaccard(map_clade, true_clade), "true_clade_post": true_prob,
            "map_clade": frozenset(map_clade), "rate_ratio": float(rate_ratio),
            "seconds": dt, "workdir": wd}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def _scphytr_rate_ratio(res, true_clade):
    """scPhyTr's clade/background rate ratio for the shift best matching the true clade."""
    rates = res["fit"].params.get("rates")
    if not rates or len(rates) < 2 or not res["shifts"]:
        return 1.0
    best = max(res["shifts"], key=lambda nd: jaccard(nd.get_leaf_names(), true_clade))
    return float(rates[res["regimes"][best]] / max(rates[0], 1e-9))


def main(reps=4, n_tips=50, rate_bg=1.0, ratios=(4.0, 16.0), ngen=50000, burnin=10000, seed0=0):
    import pandas as pd
    rows = []
    for ratio in ratios:
        for r in range(reps):
            tree, values, true_clade = simulate(n_tips, rate_bg, rate_bg * ratio, seed=seed0 + r)
            res = detect_rate_shifts(tree, values, max_shifts=4, criterion="bic")
            sp = run_scphytr(tree, values, true_clade)
            sp["rate_ratio"] = _scphytr_rate_ratio(res, true_clade)
            rb = run_revbayes(tree, values, true_clade, ngen=ngen, burnin=burnin)
            row = {"ratio": ratio, "rep": r,
                   **{f"sp_{k}": v for k, v in sp.items() if k != "method"}}
            row.update({f"rb_{k}": v for k, v in rb.items() if k not in ("method", "workdir")})
            rows.append(row)
            print(f"ratio {ratio:>4.0f}x rep{r}: scPhyTr Jac={sp['jaccard']:.2f} "
                  f"ratio={sp['rate_ratio']:.1f} ({sp['seconds']*1e3:.0f} ms)  |  "
                  f"RevBayes true-clade post={rb['true_clade_post']:.2f} "
                  f"rate-ratio={rb['rate_ratio']:.1f} ({rb['seconds']:.0f} s)")
    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT, "revbayes_rate_shifts.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")

    print("\n========== scPhyTr vs RevBayes: clade rate-shift detection ==========")
    print(f"setup: {n_tips}-tip trees, {reps} reps per shift magnitude; one true clade shift.")
    for ratio in ratios:
        d = df[df["ratio"] == ratio]
        sp_loc = (d["sp_jaccard"] >= 0.8).mean()
        rb_loc = (d["rb_true_clade_post"] >= 0.5).mean()
        print(f"  {ratio:>4.0f}x shift:  localize  scPhyTr {sp_loc:.0%} / RevBayes {rb_loc:.0%} "
              f"(post {d['rb_true_clade_post'].mean():.2f}) | "
              f"rate-ratio  scPhyTr {d['sp_rate_ratio'].mean():.1f}x / "
              f"RevBayes {d['rb_rate_ratio'].mean():.1f}x")
    sp_s, rb_s = df["sp_seconds"].mean(), df["rb_seconds"].mean()
    print(f"runtime/dataset: scPhyTr {sp_s*1e3:.0f} ms vs RevBayes {rb_s:.0f} s "
          f"(~{rb_s/max(sp_s,1e-9):.0f}x). Trade-off: fast penalized-likelihood point estimate "
          f"vs full posterior (per-branch shift probs + rate distributions).")
    return df


def figure(csv=None):
    """Plot localization and rate-magnitude recovery from the saved sweep CSV."""
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.read_csv(csv or os.path.join(OUT, "revbayes_rate_shifts.csv"))
    ratios = sorted(df["ratio"].unique())
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    # (A) localization rate vs shift magnitude
    x = np.arange(len(ratios)); w = 0.38
    sp = [(df[df.ratio == rr]["sp_jaccard"] >= 0.8).mean() for rr in ratios]
    rb = [(df[df.ratio == rr]["rb_true_clade_post"] >= 0.5).mean() for rr in ratios]
    ax[0].bar(x - w/2, sp, w, color="#2c7fb8", label="scPhyTr (BIC shift)")
    ax[0].bar(x + w/2, rb, w, color="#d95f0e", label="RevBayes (post>0.5)")
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"{int(r)}x" for r in ratios])
    ax[0].set_ylabel("fraction localizing the true shift clade"); ax[0].set_ylim(0, 1.05)
    ax[0].set_xlabel("true rate-shift magnitude"); ax[0].legend(fontsize=8)
    ax[0].set_title("(A) Shift localization vs magnitude", fontsize=10)
    # (B) estimated vs true rate ratio
    for rr in ratios:
        d = df[df.ratio == rr]
        ax[1].scatter([rr]*len(d), d["sp_rate_ratio"], color="#2c7fb8", s=30, zorder=3)
        ax[1].scatter([rr]*len(d), d["rb_rate_ratio"], color="#d95f0e", marker="s", s=30, zorder=3)
    lim = max(ratios) * 1.3
    ax[1].plot([0, lim], [0, lim], "k--", lw=1, label="truth")
    ax[1].scatter([], [], color="#2c7fb8", label="scPhyTr"); ax[1].scatter([], [], color="#d95f0e", marker="s", label="RevBayes")
    ax[1].set_xlabel("true rate ratio"); ax[1].set_ylabel("estimated clade/background rate ratio")
    ax[1].set_xlim(0, lim); ax[1].set_ylim(0, lim); ax[1].legend(fontsize=8)
    ax[1].set_title("(B) scPhyTr recovers magnitude; RevBayes shrinks", fontsize=10)
    fig.tight_layout()
    figdir = os.path.join(OUT, "figures"); os.makedirs(figdir, exist_ok=True)
    out = os.path.join(figdir, "revbayes_rate_shifts.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[fig] wrote {out}")


def real_data(tumor="3726_NT_T2", n_sub=70, gene=None, ngen=60000, burnin=15000, seed=0):
    """Apply both tools to a real KP-Tracer tumor: a gene's per-cell log-expression.

    No ground truth; the question is agreement -- do scPhyTr's BIC-selected rate shift
    and RevBayes' highest-posterior shift branch point to the same clade of cells? The
    multifurcating lineage tree is resolved to a binary tree (RevBayes' BM REML model
    requires it) with floored branch lengths.
    """
    tree, leaves, Y, pos, genes = _prep_real_tree(tumor, n_sub, seed)
    g = int(np.argmax(Y.var(0))) if gene is None else list(genes).index(gene)
    gname = str(genes[g])
    vals = {nm: float(Y[pos[nm], g]) for nm in leaves}
    print(f"real data: KP-Tracer {tumor}, {len(leaves)} cells, gene {gname} "
          f"(most variable HVG)")

    res = detect_rate_shifts(tree, vals, max_shifts=3, criterion="bic")
    sp_clade = frozenset(res["shifts"][0].get_leaf_names()) if res["shifts"] else frozenset()
    print(f"  scPhyTr: {len(res['shifts'])} shift(s); "
          f"rates {[round(r,3) for r in res['fit'].params.get('rates', [])]}; "
          f"top-shift clade = {len(sp_clade)} cells")

    rb = run_revbayes(tree, vals, frozenset(), ngen=ngen, burnin=burnin)
    print(f"  RevBayes: max posterior shift prob {rb['map_post']:.2f} on a clade of "
          f"{len(rb['map_clade'])} cells")
    print(f"  agreement (Jaccard of the two shift clades): "
          f"{jaccard(sp_clade, rb['map_clade']):.2f}")
    return {"gene": gname, "n_cells": len(leaves), "scphytr_clade_size": len(sp_clade),
            "revbayes_clade_size": len(rb["map_clade"]), "revbayes_post": rb["map_post"],
            "jaccard": jaccard(sp_clade, rb["map_clade"])}


def _prep_real_tree(tumor, n_sub, seed):
    """Load a KP-Tracer tumor, subsample cells, resolve to a binary floored tree.

    Returns (tree, leaf_names, Y, pos, genes): Y is (n_sub, n_hvg) log-norm expression
    aligned to the subsample order, pos maps leaf name -> row of Y.
    """
    import anndata as ad
    from analysis.kptracer.load import load_tumor
    DATA = os.path.join(OUT, "..", "..", "data", "external", "KPTracer-Data")
    adata = ad.read_h5ad(os.path.join(DATA, "expression", "adata_processed.nt.h5ad"))
    d = load_tumor(tumor, DATA, adata=adata, n_hvg=500)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(d["leaf_names"]), size=min(n_sub, len(d["leaf_names"])),
                             replace=False))
    names = [d["leaf_names"][i] for i in idx]
    Y = d["Y"][idx]
    et = d["tree"].copy(); et.prune(names, preserve_branch_length=True)
    et.resolve_polytomy(recursive=True)                  # binary tree for RevBayes
    H = et.get_farthest_leaf()[1] or 1.0
    for nd in et.traverse():
        if not nd.is_root() and nd.dist < 1e-3 * H:
            nd.dist = 1e-3 * H
    tree = Tree(); tree.phylotree = et; tree.root = et.get_tree_root()
    leaves = et.get_leaf_names()
    pos = {nm: i for i, nm in enumerate(names)}
    return tree, leaves, Y, pos, d["genes"]


def real_discoveries(tumor="3726_NT_T2", n_sub=64, n_genes=20, min_detect=0.6,
                     ngen=50000, burnin=10000, seed=0):
    """Compare the rate-shift DISCOVERIES of scPhyTr and RevBayes across real genes.

    No ground truth: for the top variable, well-detected genes of one KP-Tracer tumor we
    run both tools on the identical tree and trait, then ask -- how many genes does each
    flag as having a clade rate shift, do they flag the *same* genes, and when both flag a
    gene do they point to the same clade of cells? Also reports recurrent clades (a subclone
    flagged for many genes).
    """
    import pandas as pd
    tree, leaves, Y, pos, genes = _prep_real_tree(tumor, n_sub, seed)
    Yt = np.array([[Y[pos[nm], g] for nm in leaves] for g in range(Y.shape[1])])  # (genes, cells)
    detect = (Yt > 0).mean(1)                          # fraction of cells expressing
    ok = np.where(detect >= min_detect)[0]
    ok = ok[np.argsort(Yt[ok].var(1))[::-1][:n_genes]]  # most variable well-detected genes
    print(f"real-data discovery comparison: KP-Tracer {tumor}, {len(leaves)} cells, "
          f"{len(ok)} genes (detected in >={min_detect:.0%} of cells)")

    rows = []
    for gi in ok:
        gname = str(genes[gi])
        vals = {nm: float(Yt[gi, k]) for k, nm in enumerate(leaves)}
        res = detect_rate_shifts(tree, vals, max_shifts=3, criterion="bic")
        if res["shifts"]:
            top = res["shifts"][0]
            sp_clade = frozenset(top.get_leaf_names())
            rates = res["fit"].params["rates"]      # shifted-clade rate vs background (regime 0)
            sp_ratio = float(rates[res["regimes"][top]] / max(rates[0], 1e-9))
            sp_ratio = min(sp_ratio, 1e3)           # guard near-zero background
        else:
            sp_clade, sp_ratio = frozenset(), 1.0
        rb = run_revbayes(tree, vals, frozenset(), ngen=ngen, burnin=burnin)
        rows.append({"gene": gname, "sp_shift": int(len(res["shifts"]) > 0),
                     "sp_clade_size": len(sp_clade), "sp_rate_ratio": sp_ratio,
                     "sp_clade": ";".join(sorted(sp_clade)),
                     "rb_maxpost": rb["map_post"], "rb_clade_size": len(rb["map_clade"]),
                     "rb_clade": ";".join(sorted(rb["map_clade"])),
                     "jaccard": jaccard(sp_clade, rb["map_clade"])})
        r = rows[-1]
        print(f"  {gname:<22} scPhyTr {'SHIFT' if r['sp_shift'] else 'none ':<5} "
              f"(clade {r['sp_clade_size']:2d}, {r['sp_rate_ratio']:.1f}x)  |  "
              f"RevBayes max-post {r['rb_maxpost']:.2f} (clade {r['rb_clade_size']:2d})  "
              f"Jac={r['jaccard']:.2f}")
    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT, "revbayes_real_discoveries.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")

    sp_hits = df["sp_shift"] == 1
    rb_hits = df["rb_maxpost"] >= 0.5
    print("\n========== real-data rate-shift DISCOVERIES ==========")
    print(f"genes flagged with a rate shift: scPhyTr {sp_hits.sum()}/{len(df)}, "
          f"RevBayes (post>=0.5) {rb_hits.sum()}/{len(df)}")
    both = sp_hits & rb_hits
    print(f"  both flag the gene: {both.sum()}; of those, mean clade Jaccard "
          f"{df.loc[both, 'jaccard'].mean() if both.any() else float('nan'):.2f}")
    print(f"  scPhyTr-only: {(sp_hits & ~rb_hits).sum()}; RevBayes-only: "
          f"{(~sp_hits & rb_hits).sum()}; neither: {(~sp_hits & ~rb_hits).sum()}")
    # recurrent clade: the scPhyTr shift clade flagged for the most genes
    from collections import Counter
    rec = Counter(df.loc[sp_hits, "sp_clade"])
    if rec:
        top_clade, cnt = rec.most_common(1)[0]
        sz = len(top_clade.split(";")) if top_clade else 0
        print(f"  most recurrent scPhyTr shift clade ({sz} cells) flagged for {cnt} genes "
              f"-- a candidate transcriptionally-accelerated subclone")
    _discoveries_figure(df)
    return df


def _discoveries_figure(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    agree = df["jaccard"] >= 0.5
    sp = df["sp_shift"] == 1
    ax.axhline(0.5, ls=":", color="grey", lw=1)
    ax.scatter(df.loc[sp & agree, "sp_rate_ratio"], df.loc[sp & agree, "rb_maxpost"],
               s=60, color="#2c7fb8", edgecolor="k", label="both flag, same clade")
    ax.scatter(df.loc[sp & ~agree, "sp_rate_ratio"], df.loc[sp & ~agree, "rb_maxpost"],
               s=60, color="#d95f0e", edgecolor="k", label="scPhyTr flags, RevBayes/clade differ")
    ax.scatter(df.loc[~sp, "sp_rate_ratio"], df.loc[~sp, "rb_maxpost"],
               s=40, color="lightgrey", edgecolor="k", label="scPhyTr: no shift")
    ax.set_xscale("log")
    ax.set_xlabel("scPhyTr clade/background rate ratio (1 = no shift)")
    ax.set_ylabel("RevBayes max per-branch posterior shift prob")
    ax.set_title("Rate-shift discoveries on real KP-Tracer genes", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    figdir = os.path.join(OUT, "figures"); os.makedirs(figdir, exist_ok=True)
    out = os.path.join(figdir, "revbayes_real_discoveries.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[fig] wrote {out}")


if __name__ == "__main__":
    import sys
    if "discoveries" in sys.argv:
        real_discoveries()
    elif "real" in sys.argv:
        real_data()
    else:
        main()
