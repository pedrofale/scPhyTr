"""scPhyTr vs RevBayes: covariate-associated (state-dependent) BM rates, head-to-head.

A discrete covariate (a ``clone`` / spatial ``niche`` label) partitions the branches into
states, and the diffusion rate sigma^2 is allowed to differ between states. Both tools fit the
SAME state-dependent multi-rate Brownian motion on the SAME branch painting -- the difference is
the inference:

  * scPhyTr  -- maximum likelihood: per-state rates by linear-time pruning, with a calibrated
    likelihood-ratio test vs a single global rate (``scphytr.tools.fit_covariate_rates``).
    One fit, milliseconds.
  * RevBayes -- Bayesian MCMC: a lognormal-prior rate per state, posterior mean rate per state
    (``dnPhyloBrownianREML`` with branch rates indexed by the fixed state painting). Full
    posterior, but an MCMC per dataset.

We simulate a continuous trait under known per-state rates (a ``fast`` clade against a ``slow``
background), give both tools the identical tree, trait, and covariate, and compare the recovered
per-state rates and runtime. A null condition (a homogeneous-rate trait) checks that neither
manufactures a spurious rate difference.

To paint RevBayes' branches by the same parsimony reconstruction scPhyTr uses, we first run a
trivial RevBayes pass that dumps each branch's descendant taxa, match those clades to scPhyTr's
reconstructed regimes, then run the state-dependent MCMC with that fixed painting.

Requires the RevBayes binary; set ``$RB`` (default: the unpacked mac-arm64 build).
"""
import os
import re
import time
import shutil
import subprocess
import tempfile

import numpy as np
from ete3 import Tree as ETree

from scphytr.utils.tree import Tree
from scphytr.utils.pruning import paint_regimes
from scphytr.tools import fit_covariate_rates, reconstruct_states

RB = os.environ.get("RB") or shutil.which("rb") or os.path.expanduser(
    "~/opt/revbayes-v1.4.0/bin/rb")
OUT = os.path.dirname(__file__)


# --------------------------------------------------------------------------- #
# Simulation: a clade-structured covariate with state-dependent rates
# --------------------------------------------------------------------------- #

def simulate(n_tips=50, rate_slow=1.0, rate_fast=8.0, seed=0):
    """Tree + a 2-state covariate (one 'fast' clade) + a trait diffusing at the state rate."""
    rng = np.random.default_rng(seed)
    et = ETree(); et.populate(n_tips, random_branches=True)
    for nd in et.traverse():
        if not nd.is_root():
            nd.dist = round(max(float(nd.dist), 0.05), 4)
    for i, l in enumerate(et.get_leaves()):
        l.name = f"t{i}"
    tw = Tree(); tw.phylotree = et; tw.root = et.get_tree_root()
    internals = [nd for nd in et.traverse() if not nd.is_leaf() and not nd.is_root()]
    clade = max(internals, key=lambda c: min(len(c.get_leaves()), n_tips - len(c.get_leaves())))
    regimes, _ = paint_regimes(tw, [clade])
    rates = {0: rate_slow, 1: rate_fast}
    values = {}

    def desc(nd, pv):
        v = 0.0 if nd is tw.root else pv + rng.normal(0.0, np.sqrt(rates[regimes[nd]] * nd.dist))
        if nd.is_leaf():
            values[nd.name] = v
        for c in nd.children:
            desc(c, v)
    desc(tw.root, 0.0)
    fast = set(clade.get_leaf_names())
    labels = {nm: ("fast" if nm in fast else "slow") for nm in et.get_leaf_names()}
    return tw, values, labels, {"slow": rate_slow, "fast": rate_fast}


def simulate_null(n_tips=50, sigma2=1.0, seed=0):
    """Homogeneous-rate trait with the SAME clade covariate -- the false-positive control."""
    tw, _, labels, _ = simulate(n_tips, sigma2, sigma2, seed)
    rng = np.random.default_rng(10_000 + seed)
    values = {}

    def desc(nd, pv):
        v = 0.0 if nd is tw.root else pv + rng.normal(0.0, np.sqrt(sigma2 * nd.dist))
        if nd.is_leaf():
            values[nd.name] = v
        for c in nd.children:
            desc(c, v)
    desc(tw.root, 0.0)
    return tw, values, labels


# --------------------------------------------------------------------------- #
# scPhyTr
# --------------------------------------------------------------------------- #

def run_scphytr(tree, values, labels):
    t0 = time.time()
    r = fit_covariate_rates(tree, values, labels)
    dt = time.time() - t0
    return {"method": "scphytr", "rates": r["rates"], "rate_ratio": r["rate_ratio"],
            "p": r["p"], "fastest": r["fastest_state"], "seconds": dt}


# --------------------------------------------------------------------------- #
# RevBayes: fixed-painting state-dependent BM
# --------------------------------------------------------------------------- #

_DUMP_TEMPLATE = """\
tree <- readTrees("{nwk}")[1]
nn = tree.nnodes()
nb = nn - 1
write("idx", "taxa", filename="{clades}", separator="\\t")
write("\\n", filename="{clades}", append=TRUE)
for (i in 1:nb) {{
    write(i, tree.getDescendantTaxa(i), filename="{clades}", separator="\\t", append=TRUE)
    write("\\n", filename="{clades}", append=TRUE)
}}
q()
"""

_STATEDEP_TEMPLATE = """\
trait <- readContinuousCharacterData("{nex}")
tree  <- readTrees("{nwk}")[1]
nn = tree.nnodes()
nb = nn - 1
moves = VectorMoves()
NS = {nstate}
for (s in 1:NS) {{
    log_r[s] ~ dnUniform(-10, 5)
    moves.append( mvSlide(log_r[s], weight=2) )
    state_rate[s] := exp(log_r[s])
}}
bs <- v({bs_csv})
for (i in 1:nb) {{ branch_rate[i] := state_rate[bs[i]] }}
seq ~ dnPhyloBrownianREML(tree, branchRates=branch_rate, nSites=1)
seq.clamp(trait)
mymodel = model(state_rate)
monitors = VectorMonitors()
monitors.append( mnModel(filename="{log}", printgen=20) )
mc = mcmc(mymodel, monitors, moves)
mc.burnin(generations={burnin}, tuningInterval=200)
mc.run(generations={ngen})
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


def _dump_clades(nwk, wd, rb):
    clades = os.path.join(wd, "clades.txt")
    rev = os.path.join(wd, "dump.Rev")
    with open(rev, "w") as f:
        f.write(_DUMP_TEMPLATE.format(nwk=nwk, clades=clades))
    subprocess.run([rb, rev], check=True, capture_output=True, text=True)
    idx2clade = {}
    with open(clades) as f:
        next(f, None)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            taxa = re.findall(r"[\w.]+", parts[1].strip())
            idx2clade[int(parts[0])] = frozenset(taxa)
    return idx2clade


def _branch_states(tree, labels, idx2clade):
    """Align scPhyTr's parsimony regimes to RevBayes' branch indices (1-based state ids)."""
    regimes, n_regimes, state_names = reconstruct_states(tree, labels)
    lookup = {frozenset(nd.get_leaf_names()): regimes[nd]
              for nd in tree.root.traverse() if nd is not tree.root}
    nb = len(idx2clade)
    bs = []
    for i in range(1, nb + 1):
        reg = lookup.get(idx2clade[i], 0)
        bs.append(reg + 1)                       # RevBayes states are 1-based
    return bs, n_regimes, state_names


def run_revbayes(tree, values, labels, ngen=20000, burnin=3000, rb=RB, keep_dir=None):
    if not os.path.exists(rb):
        raise FileNotFoundError(f"RevBayes binary not found at {rb}; set $RB.")
    wd = keep_dir or tempfile.mkdtemp(prefix="rb_covrate_")
    os.makedirs(wd, exist_ok=True)
    nwk, nex = _write_inputs(tree, values, wd)
    idx2clade = _dump_clades(nwk, wd, rb)
    bs, nstate, state_names = _branch_states(tree, labels, idx2clade)
    logf = os.path.join(wd, "statedep.log")
    rev = os.path.join(wd, "statedep.Rev")
    with open(rev, "w") as f:
        f.write(_STATEDEP_TEMPLATE.format(nex=nex, nwk=nwk, log=logf, nstate=nstate,
                                          bs_csv=",".join(map(str, bs)),
                                          burnin=burnin, ngen=ngen))
    t0 = time.time()
    subprocess.run([rb, rev], check=True, capture_output=True, text=True)
    dt = time.time() - t0

    import csv
    with open(logf) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    rates = {}
    for s in range(1, nstate + 1):
        col = f"state_rate[{s}]"
        rates[state_names[s - 1]] = float(np.mean([float(r[col]) for r in rows]))
    ratio = max(rates.values()) / max(min(rates.values()), 1e-12)
    return {"method": "revbayes", "rates": rates, "rate_ratio": ratio,
            "fastest": max(rates, key=rates.get), "seconds": dt, "workdir": wd}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def main(reps=4, n_tips=50, rate_slow=1.0, ratios=(4.0, 16.0), ngen=30000, burnin=5000, seed0=0):
    import pandas as pd
    rows = []
    for ratio in ratios:
        for r in range(reps):
            tree, values, labels, true_rates = simulate(n_tips, rate_slow, rate_slow * ratio,
                                                         seed=seed0 + r)
            sp = run_scphytr(tree, values, labels)
            rb = run_revbayes(tree, values, labels, ngen=ngen, burnin=burnin)
            rows.append({
                "ratio": ratio, "rep": r,
                "true_slow": true_rates["slow"], "true_fast": true_rates["fast"],
                "sp_slow": sp["rates"]["slow"], "sp_fast": sp["rates"]["fast"],
                "sp_ratio": sp["rate_ratio"], "sp_p": sp["p"], "sp_seconds": sp["seconds"],
                "rb_slow": rb["rates"]["slow"], "rb_fast": rb["rates"]["fast"],
                "rb_ratio": rb["rate_ratio"], "rb_seconds": rb["seconds"]})
            row = rows[-1]
            print(f"ratio {ratio:>4.0f}x rep{r}: scPhyTr fast/slow={row['sp_fast']:.1f}/{row['sp_slow']:.2f} "
                  f"(ratio {row['sp_ratio']:.1f}, p={row['sp_p']:.1e}, {row['sp_seconds']*1e3:.0f} ms)  |  "
                  f"RevBayes fast/slow={row['rb_fast']:.1f}/{row['rb_slow']:.2f} "
                  f"(ratio {row['rb_ratio']:.1f}, {row['rb_seconds']:.0f} s)")
    # null false-positive control
    null_rows = []
    for r in range(reps):
        tree, values, labels = simulate_null(n_tips, rate_slow, seed=seed0 + r)
        sp = run_scphytr(tree, values, labels)
        null_rows.append({"rep": r, "sp_p": sp["p"], "sp_ratio": sp["rate_ratio"]})
        print(f"null rep{r}: scPhyTr ratio={sp['rate_ratio']:.1f}  p={sp['p']:.2f}")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT, "revbayes_covariate_rates.csv")
    df.to_csv(out_csv, index=False)
    pd.DataFrame(null_rows).to_csv(os.path.join(OUT, "revbayes_covariate_rates_null.csv"), index=False)
    print(f"\nwrote {out_csv}")

    print("\n========== scPhyTr vs RevBayes: covariate-associated rate recovery ==========")
    print(f"setup: {n_tips}-tip trees, {reps} reps per magnitude; a 2-state covariate (one 'fast' clade).")
    for ratio in ratios:
        d = df[df["ratio"] == ratio]
        print(f"  {ratio:>4.0f}x:  true ratio {ratio:.0f}  |  "
              f"scPhyTr {d['sp_ratio'].mean():.1f}x (p {d['sp_p'].median():.1e})  /  "
              f"RevBayes {d['rb_ratio'].mean():.1f}x")
    nd = pd.DataFrame(null_rows)
    print(f"null (homogeneous rate): scPhyTr false-positive rate (p<0.05) = {(nd['sp_p'] < 0.05).mean():.0%}")
    sp_s, rb_s = df["sp_seconds"].mean(), df["rb_seconds"].mean()
    print(f"runtime/dataset: scPhyTr {sp_s*1e3:.0f} ms vs RevBayes {rb_s:.0f} s "
          f"(~{rb_s/max(sp_s,1e-9):.0f}x).")
    return df


def figure(csv=None):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.read_csv(csv or os.path.join(OUT, "revbayes_covariate_rates.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    # (A) recovered fast-state rate vs true
    ax[0].scatter(df["true_fast"], df["sp_fast"], color="#2c7fb8", s=40, label="scPhyTr", zorder=3)
    ax[0].scatter(df["true_fast"], df["rb_fast"], color="#d95f0e", marker="s", s=40,
                  label="RevBayes", zorder=3)
    lim = df["true_fast"].max() * 1.25
    ax[0].plot([0, lim], [0, lim], "k--", lw=1, label="truth")
    ax[0].set_xlabel("true fast-state rate"); ax[0].set_ylabel("estimated fast-state rate")
    ax[0].set_title("(A) Per-state rate recovery", fontsize=10); ax[0].legend(fontsize=8)
    # (B) recovered rate ratio vs true
    ratios = sorted(df["ratio"].unique())
    for rr in ratios:
        d = df[df.ratio == rr]
        ax[1].scatter([rr]*len(d), d["sp_ratio"], color="#2c7fb8", s=30, zorder=3)
        ax[1].scatter([rr]*len(d), d["rb_ratio"], color="#d95f0e", marker="s", s=30, zorder=3)
    lim = max(ratios) * 1.4
    ax[1].plot([0, lim], [0, lim], "k--", lw=1, label="truth")
    ax[1].scatter([], [], color="#2c7fb8", label="scPhyTr")
    ax[1].scatter([], [], color="#d95f0e", marker="s", label="RevBayes")
    ax[1].set_xlabel("true rate ratio"); ax[1].set_ylabel("estimated state rate ratio")
    ax[1].set_xlim(0, lim); ax[1].set_ylim(0, lim)
    ax[1].set_title("(B) Fast/slow ratio recovery", fontsize=10); ax[1].legend(fontsize=8)
    fig.tight_layout()
    figdir = os.path.join(OUT, "figures"); os.makedirs(figdir, exist_ok=True)
    out = os.path.join(figdir, "revbayes_covariate_rates.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[fig] wrote {out}")


if __name__ == "__main__":
    main()
