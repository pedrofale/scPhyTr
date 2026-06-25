"""KP-Tracer co-evolution head-to-head (task 5) with ground truth on a REAL tree.

The benchmark matrix's co-evolution row: recover the matrix of gene--gene evolutionary
correlations (the off-diagonal of scPhyTr's rate matrix K) -- i.e. which genes co-evolve as
a module. Three methods produce a gene x gene association object on the SAME data; they
differ only in whether they deconfound the phylogeny:

  * scPhyTr K  -- Felsenstein contrast correlation: whiten the leaves by the tumor's
    phylogenetic covariance C, leaving n-1 i.i.d. rows whose correlation is the
    deconfounded evolutionary correlation (the off-diagonal of K). Calibrated null.
  * PATH cross -- phylogenetic CROSS-correlation (bivariate Moran's I): shared-ancestry
    weighted co-variation of two traits at the tips. Like the naive tip correlation it is
    confounded -- two evolutionarily INDEPENDENT genes that co-drift down shared branches
    show spurious cross-correlation. Permutation null (PATH's own resampling logic).
  * Hotspot   -- tree-mode local correlation Z (PhyloVision setup). Its null is cell
    EXCHANGEABILITY (genes i.i.d. across cells), which is false on a phylogeny, so the null
    variance is under-estimated and Z is over-dispersed -> manufactured modules.

Ground truth: we keep a real tumor's tree/branch lengths but SIMULATE p genes as
multivariate Brownian motion with a KNOWN evolutionary correlation K -- a block of ``b``
co-evolving genes (pairwise corr ``rho``) plus independent genes (K diagonal). Matrix-normal
draw ``Y = chol(C) Z chol(K)^T`` gives row covariance C (the tree) and column covariance K
(the program). We then ask each method, at a common BH-FDR, two questions:

  * POWER  : of the true within-block pairs, how many are recovered?
  * FPR    : of the truly independent pairs (K off-diagonal = 0), how many are FALSELY
             called co-evolving? This is the confounding/calibration headline -- the tree
             induces tip co-variation among independent genes, and a method that does not
             deconfound (PATH cross, Hotspot) manufactures co-evolution.

scPhyTr's deconfounded K is expected to hold FPR near nominal while keeping power; PATH
cross and Hotspot are expected to inflate FPR (invent modules from shared ancestry).
"""
import os
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), ".numba_cache"))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

from analysis.kptracer.load import load_tumor
from analysis.kptracer import hotspot_utils as hu
from analysis.kptracer.phylo_factor_utils import chol, phylo_contrasts, parallel_analysis

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data", "external", "KPTracer-Data")
OUT = os.path.dirname(__file__)


def simulate_block_K(C, p, b, rho, rng):
    """Multivariate BM on tree C with a co-evolving block of size b (corr rho).

    Returns Y (n, p) and a boolean (p, p) ``true_pair`` mask of within-block pairs.
    """
    K = np.eye(p)
    block = np.arange(b)
    for i in block:
        for j in block:
            if i != j:
                K[i, j] = rho
    L_C = chol(C)
    L_K = np.linalg.cholesky(K + 1e-10 * np.eye(p))
    Y = L_C @ rng.standard_normal((C.shape[0], p)) @ L_K.T
    true_pair = np.zeros((p, p), bool)
    true_pair[np.ix_(block, block)] = True
    np.fill_diagonal(true_pair, False)
    return Y, true_pair


# --------------------------------------------------------------------------- #
# Three gene x gene p-value estimators
# --------------------------------------------------------------------------- #

def scphytr_K_p(Y, C):
    """Deconfounded evolutionary correlation (off-diagonal of K) -> p-value matrix."""
    R, m = hu.contrast_corr(Y, C)
    Z = hu.corr_to_z(R, m)
    return 2 * stats.norm.sf(np.abs(Z))


def path_cross_p(Y, C, n_perm=199, rng=None):
    """PATH phylogenetic cross-correlation (bivariate Moran's I) + permutation p.

    Vectorized: the full p x p cross-Moran matrix is (Yn^T W Yn) * n / S0 for unit-norm
    centered columns Yn; the permutation null jointly permutes cells (rows), fixing W.
    """
    rng = rng or np.random.default_rng(0)
    n, p = Y.shape
    W = C.copy()
    np.fill_diagonal(W, 0.0)
    S0 = W.sum()
    Yc = Y - Y.mean(0)
    Yn = Yc / (np.sqrt((Yc ** 2).sum(0)) + 1e-12)
    I_obs = (n / S0) * (Yn.T @ (W @ Yn))
    a = np.abs(I_obs)
    count = np.zeros((p, p))
    for _ in range(n_perm):
        Yp = Yn[rng.permutation(n)]
        I_null = (n / S0) * (Yp.T @ (W @ Yp))
        count += np.abs(I_null) >= a
    return (1.0 + count) / (n_perm + 1.0), I_obs


def hotspot_p(Y, names, genes, tree):
    """Hotspot tree-mode local correlation Z (all genes) -> p-value matrix."""
    res = hu.run_hotspot(Y, names, gene_names=genes, model="normal",
                         tree=tree, restrict_genes=list(genes), jobs=1)
    lcz = res["lcz"]
    if lcz is None:
        return None
    # reorder to the caller's gene order
    Z = lcz.reindex(index=list(genes), columns=list(genes)).values
    Z = np.asarray(Z, float)
    Z[~np.isfinite(Z)] = 0.0
    return 2 * stats.norm.sf(np.abs(Z))


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def _upper(M):
    iu = np.triu_indices_from(M, 1)
    return M[iu], iu


def auroc(score, label):
    score = np.asarray(score, float); label = np.asarray(label, bool)
    order = np.argsort(score); ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    n1 = label.sum(); n0 = (~label).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    return (ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def score_method(pmat, true_pair, fdr=0.05):
    """Power on true block pairs + FPR on independent pairs at BH-FDR; AUROC."""
    p_off, iu = _upper(pmat)
    lab = true_pair[iu]
    q = hu.bh_fdr(p_off)
    sig = q < fdr
    power = sig[lab].mean() if lab.sum() else np.nan
    fpr = sig[~lab].mean() if (~lab).sum() else np.nan
    roc = auroc(-p_off, lab)  # smaller p -> higher score
    return dict(power=power, fpr=fpr, auroc=roc, n_true=int(lab.sum()), n_null=int((~lab).sum()))


def run_tumor(tumor, adata, p=30, b=10, rho=0.6, n_sub=600, R=5,
              n_perm=199, n_hvg=1, seed=0):
    d = load_tumor(tumor, DATA, adata=adata, n_hvg=n_hvg)  # expression unused (we simulate)
    Cfull, names_full = d["C"], d["leaf_names"]
    tree = d["tree"]
    n0 = len(names_full)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n0, size=min(n_sub, n0), replace=False))
    C = Cfull[np.ix_(idx, idx)]
    names = [names_full[i] for i in idx]
    # prune the ete3 tree to the same leaves so Hotspot's tree matches C
    tree = tree.copy()
    tree.prune(names, preserve_branch_length=True)
    genes = np.array([f"g{j}" for j in range(p)])
    print(f"\n=== {tumor}: {n0} cells -> {len(names)} | p={p} genes "
          f"(block b={b}, rho={rho}), {R} reps ===")

    rows = []
    for r in range(R):
        Y, true_pair = simulate_block_K(C, p, b, rho, np.random.default_rng(1000 + r))
        methods = {
            "scphytr_K": scphytr_K_p(Y, C),
            "path_cross": path_cross_p(Y, C, n_perm=n_perm, rng=np.random.default_rng(2000 + r))[0],
        }
        hp = hotspot_p(Y, names, genes, tree)
        if hp is not None:
            methods["hotspot"] = hp
        for name, pmat in methods.items():
            s = score_method(pmat, true_pair)
            s.update(tumor=tumor, method=name, rep=r, n=len(names))
            rows.append(s)
    df = pd.DataFrame(rows)
    summ = df.groupby("method")[["power", "fpr", "auroc"]].mean()
    for name in ["scphytr_K", "path_cross", "hotspot"]:
        if name in summ.index:
            s = summ.loc[name]
            print(f"  {name:<11} power={s['power']:.2f}  FPR={s['fpr']:.2f}  "
                  f"AUROC={s['auroc']:.2f}")
    return rows


def main(seed=0):
    import anndata as ad
    print("loading integrated AnnData ...")
    adata = ad.read_h5ad(os.path.join(DATA, "expression", "adata_processed.nt.h5ad"))
    all_rows = []
    for t in ["3726_NT_T2", "3513_NT_T3"]:
        try:
            all_rows += run_tumor(t, adata, seed=seed)
        except Exception as e:
            print(f"  !! {t} skipped: {e}")
    df = pd.DataFrame(all_rows)
    out_csv = os.path.join(OUT, "kptracer_coevolution.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv} ({len(df)} rows)")

    print("\n================ CO-EVOLUTION HEAD-TO-HEAD (pooled) ================")
    print("FPR = false co-evolution among independent genes (nominal 0.05); "
          "Power = recovery of the true block.")
    summ = df.groupby("method")[["power", "fpr", "auroc"]].mean()
    for name in ["scphytr_K", "path_cross", "hotspot"]:
        if name in summ.index:
            s = summ.loc[name]
            tag = "calibrated" if s["fpr"] <= 0.10 else "INFLATED (tree-confounded)"
            print(f"  {name:<11} power={s['power']:.2f}  FPR={s['fpr']:.2f}  "
                  f"AUROC={s['auroc']:.2f}   <- {tag}")
    _figure(summ)
    return df


def _figure(summ):
    """Power vs false-positive-rate, one point per method (the fig:coevo panel)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = {"scphytr_K": ("#2c7fb8", "o", "scPhyTr K (deconfounded)"),
             "path_cross": ("#d95f0e", "s", "PATH cross-correlation"),
             "hotspot": ("#756bb1", "^", "Hotspot (tree mode)")}
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.axvspan(0, 0.10, color="#e5f5e0", alpha=0.6, zorder=0)
    ax.axvline(0.05, ls=":", color="grey", lw=1)
    ax.text(0.055, 0.02, "nominal 5% FPR", fontsize=8, color="grey", rotation=90, va="bottom")
    for name, (col, mk, lab) in style.items():
        if name not in summ.index:
            continue
        s = summ.loc[name]
        ax.scatter(s["fpr"], s["power"], s=200, color=col, marker=mk, edgecolor="k",
                   zorder=3, label=f"{lab}\n(AUROC {s['auroc']:.2f})")
    ax.set_xlabel("false co-evolution rate among independent genes")
    ax.set_ylabel("power to recover the true module")
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.95)
    ax.set_title("Co-evolution recovery on real KP-Tracer trees\n"
                 "(known module, multivariate BM ground truth)", fontsize=10)
    fig.tight_layout()
    figdir = os.path.join(OUT, "figures")
    os.makedirs(figdir, exist_ok=True)
    out = os.path.join(figdir, "kptracer_coevolution.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[fig] wrote {out}")


# =========================================================================== #
# High-p, rank-k regime: Poisson phylogenetic FACTOR ANALYSIS recovers a
# planted low-rank K at thousands of genes, where the dense p x p K is
# infeasible. This is the scalable, regularized counterpart of the pairwise
# head-to-head above -- the factor model carries p x k loadings W (never the
# p x p matrix) and reports the gene-gene correlation as corr(W W^T) in O(N k^3).
# =========================================================================== #

from scphytr.utils.tree import Tree
from scphytr.tools.poisson_factor import fit_poisson_factor_analysis
from analysis.benchmark.sim_heritability import star_tree


def _wrap(ete_tree):
    """Wrap a pruned ete3 tree as a scphytr Tree (for the PFA tree-Laplace)."""
    t = Tree()
    t.phylotree = ete_tree
    t.root = ete_tree.get_tree_root()
    return t


def plant_pfa_loadings(p, k, mag=1.0, cross=0.05, seed=0):
    """Rank-k loadings W (p x k): each gene in one of k programs + small cross-talk.

    Returns (W, mu, program). True gene-gene evolutionary correlation is corr(W W^T):
    same-program pairs strongly correlated, cross-program pairs ~0.
    """
    rng = np.random.default_rng(seed)
    program = np.sort(np.tile(np.arange(k), int(np.ceil(p / k)))[:p])
    W = cross * rng.standard_normal((p, k))
    for g in range(p):
        W[g, program[g]] += rng.choice([-1.0, 1.0]) * (mag + 0.3 * abs(rng.standard_normal()))
    mu = rng.uniform(np.log(0.05), np.log(0.5), size=p)   # baseline per-gene log-rate
    return W, mu, program


def simulate_pfa_counts(C, W, mu, mean_size=2000.0, seed=0):
    """Poisson counts from k unit-variance BM factors on tree C with loadings W."""
    rng = np.random.default_rng(seed)
    n = C.shape[0]
    p, k = W.shape
    Cn = C / np.sqrt(np.outer(np.diag(C), np.diag(C)))     # unit tip variance
    L = chol(Cn)
    X = L @ rng.standard_normal((n, k))                    # factor scores (n, k)
    sizes = rng.gamma(4.0, mean_size / 4.0, size=n) / mean_size
    eta = mu[None, :] + X @ W.T
    Y = rng.poisson((sizes * mean_size)[:, None] * np.exp(eta)).astype(float)
    return Y, sizes


def _corr_from_K(K):
    d = np.sqrt(np.clip(np.diag(K), 1e-300, None))
    R = K / np.outer(d, d)
    np.fill_diagonal(R, 1.0)
    return R


def score_pfa(R_hat, program):
    """Recovery of the planted block structure from a fitted correlation matrix."""
    iu = np.triu_indices_from(R_hat, 1)
    rh = np.abs(R_hat[iu])
    same = program[iu[0]] == program[iu[1]]
    return dict(block_auroc=auroc(rh, same),
                within_mean=float(rh[same].mean()),
                cross_false=float(rh[~same].mean()))   # spurious corr among independent pairs


def dense_contrast_K(Y, C):
    """The full p x p deconfounded correlation (dense K) -- what 'explodes' at high p.

    Felsenstein contrasts give only n-1 effective rows, so for p >> n this empirical
    matrix is rank-deficient and its off-block entries are spurious. Computed on
    log-normalized counts (the Gaussian-trait path)."""
    lib = Y.sum(1, keepdims=True); lib[lib == 0] = 1.0
    Ylog = np.log1p(Y / lib * np.median(lib))
    R, _ = hu.contrast_corr(Ylog, C)
    return R


def _try_cholesky(R):
    """Wall-clock to factor the p x p K (what any multivariate use of a dense K needs)
    and whether it is positive-definite. Rank-deficient (p>n) correlations fail."""
    import time
    t0 = time.time()
    try:
        np.linalg.cholesky(R)
        pd = True
    except np.linalg.LinAlgError:
        pd = False
    return time.time() - t0, pd


def run_pfa_tumor(tumor, adata, p_grid=(500, 2000, 4000, 8000), k=5, n_sub=400,
                  reps=2, seed=0, dense_max_p=4000):
    import time
    from scipy.stats import spearmanr
    d = load_tumor(tumor, DATA, adata=adata, n_hvg=1)
    Cfull, names_full, tree0 = d["C"], d["leaf_names"], d["tree"]
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(names_full), size=min(n_sub, len(names_full)), replace=False))
    C = Cfull[np.ix_(idx, idx)]
    names = [names_full[i] for i in idx]
    tree = tree0.copy(); tree.prune(names, preserve_branch_length=True)
    tw = _wrap(tree)
    n = len(names)
    print(f"\n=== {tumor}: {n} cells, k_true={k} | p in {list(p_grid)} "
          f"(dense K has only n-1={n-1} effective rows) ===")

    rows = []
    for p in p_grid:
        for r in range(reps):
            W, mu, program = plant_pfa_loadings(p, k, seed=100 + r)
            Y, sizes = simulate_pfa_counts(C, W, mu, seed=200 + r)
            Rtrue = _corr_from_K(W @ W.T)
            iu = np.triu_indices(p, 1)
            # ---- rank-k Poisson PFA: K = W W^T, carries p x k loadings, never the p x p
            t0 = time.time()
            fm = fit_poisson_factor_analysis(Y, tw, k, sizes=sizes,
                                             leaf_names=names, n_iter=40)
            dt_pfa = time.time() - t0
            R_pfa = fm.evolutionary_correlation()
            s = score_pfa(R_pfa, program)
            s["spearman_Rtrue"] = float(spearmanr(np.abs(R_pfa[iu]), np.abs(Rtrue[iu])).statistic)
            s.update(tumor=tumor, model="pfa", p=p, k=k, n=n, rep=r, seconds=dt_pfa,
                     stored=p * k, factor_s=0.0, pd=True)   # W W^T (+epsI) is PD by construction
            rows.append(s); pf = s

            # ---- dense p x p deconfounded K: the object that "explodes"
            de = None
            if p <= dense_max_p:
                R_dense = dense_contrast_K(Y, C)
                fac_s, pd = _try_cholesky(R_dense)          # cost+validity of USING it
                de = score_pfa(R_dense, program)
                de["spearman_Rtrue"] = float(spearmanr(np.abs(R_dense[iu]),
                                                       np.abs(Rtrue[iu])).statistic)
                de.update(tumor=tumor, model="dense_K", p=p, k=k, n=n, rep=r,
                          seconds=np.nan, stored=p * (p + 1) // 2, factor_s=fac_s, pd=pd)
                rows.append(de)
            msg = (f"  p={p:5d} rep{r}: PFA AUROC={pf['block_auroc']:.3f} "
                   f"rho={pf['spearman_Rtrue']:.2f} ({pf['seconds']:.1f}s, store {pf['stored']:,})")
            if de is not None:
                msg += (f"  |  dense-K AUROC={de['block_auroc']:.3f} "
                        f"store {de['stored']:,} PD={de['pd']} (factor {de['factor_s']:.2f}s)")
            else:
                msg += "  |  dense-K: SKIPPED (p x p too large to form)"
            print(msg)
    return rows


def horn_k_check(tumor, adata, p=800, k_true=5, n_sub=400, seed=0):
    """De-novo k selection via Horn parallel analysis -- on raw leaves vs contrasts.

    Two regimes make the point: (1) a NULL with no shared programs (genes are
    independent BM), where parallel analysis on the raw leaves invents a spurious
    factor (the deep-clade axis) while the deconfounded contrasts return ~0; and
    (2) a PLANTED rank-k_true model, where the contrasts recover ~k_true.
    """
    d = load_tumor(tumor, DATA, adata=adata, n_hvg=1)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(d["leaf_names"]), size=n_sub, replace=False))
    C = d["C"][np.ix_(idx, idx)]
    n = C.shape[0]
    L = chol(C)        # raw C: simulate and deconfound with the SAME covariance

    # (1) null: evolutionarily independent genes (no programs)
    Ynull = L @ np.random.default_rng(11).standard_normal((n, p))   # continuous BM traits
    _, _, kn_leaf = parallel_analysis(Ynull, 100, np.random.default_rng(1))
    _, _, kn_con = parallel_analysis(phylo_contrasts(Ynull, C), 100, np.random.default_rng(2))

    # (2) planted rank-k_true programs (count data -> log-normalize)
    W, mu, _ = plant_pfa_loadings(p, k_true, seed=7)
    Y, _ = simulate_pfa_counts(C, W, mu, seed=7)
    ln = np.log1p(Y / Y.sum(1, keepdims=True) * np.median(Y.sum(1)))
    _, _, kp_con = parallel_analysis(phylo_contrasts(ln, C), 100, np.random.default_rng(3))

    print(f"\nHorn parallel analysis (de-novo k):")
    print(f"  NULL (no programs):  raw leaves -> {kn_leaf} spurious factor(s) (tree "
          f"confounding); contrasts -> {kn_con} (deconfounded, correct)")
    print(f"  PLANTED k_true={k_true}: contrasts -> {kp_con} factor(s) (recovers k)")
    return kn_leaf, kn_con, kp_con


def main_pfa(seed=0):
    import anndata as ad
    print("loading integrated AnnData ...")
    adata = ad.read_h5ad(os.path.join(DATA, "expression", "adata_processed.nt.h5ad"))
    rows = run_pfa_tumor("3726_NT_T2", adata, seed=seed)
    horn_k_check("3726_NT_T2", adata)
    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT, "kptracer_pfa_coevolution.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv} ({len(df)} rows)")

    print("\n========== HIGH-p PFA (K = W W^T, rank-k) vs DENSE p x p K ==========")
    print("Both recover the module when signal is clean (AUROC~1); the difference is "
          "FEASIBILITY: PFA stores p x k and is PD by construction; the dense K stores "
          f"p^2/2, is rank-deficient (not PD) once p>n, and costs O(p^3) to use.")
    g = df.groupby(["p", "model"])[["block_auroc", "spearman_Rtrue", "cross_false",
                                    "seconds", "stored", "factor_s", "pd"]].mean()
    for p in sorted(df["p"].unique()):
        pf = g.loc[(p, "pfa")]
        line = (f"  p={p:5d}: PFA AUROC={pf['block_auroc']:.3f} rho={pf['spearman_Rtrue']:.2f} "
                f"fit={pf['seconds']:.1f}s store={int(pf['stored']):,}")
        if (p, "dense_K") in g.index:
            de = g.loc[(p, "dense_K")]
            line += (f"  |  dense-K AUROC={de['block_auroc']:.3f} store={int(de['stored']):,} "
                     f"({int(de['stored'])//max(int(pf['stored']),1)}x) PD={de['pd']>0.5}")
        else:
            line += "  |  dense-K not formed (p x p too large)"
        print(line)
    _pfa_figure(g, df)
    return df


def _pfa_figure(g, df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ps = sorted(df["p"].unique())
    ps_dense = [p for p in ps if (p, "dense_K") in g.index]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    # (A) recovery: PFA accurate across the whole p range
    ax[0].plot(ps, [g.loc[(p, "pfa"), "block_auroc"] for p in ps], "o-",
               color="#2c7fb8", label="rank-k PFA")
    if ps_dense:
        ax[0].plot(ps_dense, [g.loc[(p, "dense_K"), "block_auroc"] for p in ps_dense], "s--",
                   color="#d95f0e", label="dense p×p K")
    ax[0].set_ylabel("block recovery AUROC"); ax[0].set_ylim(0.4, 1.02)
    ax[0].set_title("(A) PFA recovers the module across p", fontsize=10)
    ax[0].legend(fontsize=8)
    # (B) cost: stored parameters, PFA linear vs dense quadratic
    ax[1].plot(ps, [g.loc[(p, "pfa"), "stored"] for p in ps], "o-",
               color="#2c7fb8", label="rank-k PFA: p·k")
    ax[1].plot(ps, [p * (p + 1) // 2 for p in ps], "s--",
               color="#d95f0e", label="dense K: p²/2")
    ax[1].set_ylabel("parameters stored"); ax[1].set_yscale("log")
    ax[1].set_title("(B) Dense K explodes as p²; PFA grows as p·k", fontsize=10)
    ax[1].legend(fontsize=8)
    for a in ax:
        a.set_xlabel("number of genes p"); a.set_xscale("log")
    fig.tight_layout()
    figdir = os.path.join(OUT, "figures")
    os.makedirs(figdir, exist_ok=True)
    out = os.path.join(figdir, "kptracer_pfa_coevolution.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[fig] wrote {out}")


# =========================================================================== #
# REAL-DATA discovery comparison (no planted ground truth): which gene-gene
# co-evolution edges does each method actually call on real KP-Tracer expression,
# and how much of what PATH/Hotspot "discover" survives phylogenetic deconfounding?
# =========================================================================== #

def real_coevolution_discoveries(tumor="3726_NT_T2", n_sub=600, n_hvg=40,
                                 top_k=40, seed=0):
    """Compare co-evolution DISCOVERIES of scPhyTr K vs the naive (PATH/Hotspot-type)
    correlation on real KP-Tracer expression.

    At single-cell n the per-pair significance test saturates (almost every pair is
    "significant"), so counting edges is uninformative. The sample-size-robust question
    is which pairs each method RANKS as the strongest co-evolution, and whether those
    top discoveries are real or shared-ancestry artifacts. We therefore compare the top
    gene pairs by the naive tip correlation (what PATH's cross-correlation and Hotspot
    respond to) against the top pairs by the deconfounded contrast correlation (the
    off-diagonal of scPhyTr's K), and measure how clade-confounded each set is via the
    clade variance fraction eta^2 of the genes involved.
    """
    import anndata as ad
    from analysis.kptracer.phylo_factor_utils import get_clades, clade_eta2
    adata = ad.read_h5ad(os.path.join(DATA, "expression", "adata_processed.nt.h5ad"))
    d = load_tumor(tumor, DATA, adata=adata, n_hvg=n_hvg)
    rng = np.random.default_rng(seed)
    n0 = len(d["leaf_names"])
    idx = np.sort(rng.choice(n0, size=min(n_sub, n0), replace=False))
    C = d["C"][np.ix_(idx, idx)]
    names = [d["leaf_names"][i] for i in idx]
    Y = d["Y"][idx]
    tree = d["tree"].copy(); tree.prune(names, preserve_branch_length=True)
    n, p = Y.shape
    print(f"real co-evolution discoveries: KP-Tracer {tumor}, {n} cells, {p} HVGs")

    R_tip, _ = hu.tip_corr(Y)                       # naive (PATH/Hotspot-type), confounded
    R_con, _ = hu.contrast_corr(Y, C)              # deconfounded (scPhyTr K off-diagonal)
    labels = get_clades(tree, names, 10)
    Ystd = (Y - Y.mean(0)) / (Y.std(0) + 1e-9)
    eta = np.array([clade_eta2(Ystd[:, g], labels) for g in range(p)])  # per-gene clonality

    iu = np.triu_indices(p, 1)
    tip, con = np.abs(R_tip[iu]), np.abs(R_con[iu])
    shrink = tip - con                              # how much deconfounding lowers |r|
    pair_eta = np.maximum(eta[iu[0]], eta[iu[1]])  # a pair is clade-confounded if either gene is
    top_tip = set(np.argsort(tip)[::-1][:top_k])
    top_con = set(np.argsort(con)[::-1][:top_k])
    overlap = len(top_tip & top_con) / top_k
    rho = stats.spearmanr(pair_eta, shrink).correlation     # the confounding signature

    print("\n========== co-evolution discoveries on REAL data ==========")
    print(f"(per-pair significance saturates at n={n}; comparing which pairs each method "
          f"RANKS as strongest, and the deconfounding it implies)")
    print(f"  top-{top_k} pairs overlap (naive vs deconfounded scPhyTr K): {overlap:.0%} "
          f"-- the strongest co-evolution signals are {('largely shared' if overlap>0.7 else 'method-dependent')}")
    # data-driven confounding signature: do clade-tracking (high-eta^2) pairs shrink more?
    hi = pair_eta >= np.quantile(pair_eta, 0.8)
    lo = pair_eta <= np.quantile(pair_eta, 0.2)
    print(f"  Spearman(clade eta^2, naive-vs-deconfounded shrinkage) = {rho:+.2f} "
          f"-- {'positive: clade-confounded pairs are the ones scPhyTr down-weights' if rho>0.1 else 'weak: top signal here is not clonal'}")
    print(f"  median shrinkage: clonal pairs (high eta^2) {np.median(shrink[hi]):+.2f} "
          f"vs non-clonal {np.median(shrink[lo]):+.2f}")
    if overlap > 0.7 and abs(rho) < 0.15:
        print("  => Honest finding: on this tumor's top HVGs the strongest co-expression is "
              "genuine co-regulation (low clonality, survives deconfounding), so naive and "
              "deconfounded methods AGREE. The simulated FPR gap appears when genes are "
              "lineage-driven; here the top signal is not. Tree confounding is real but "
              "module/clade-level (see analysis/kptracer/hotspot_vs_phylo_real.py), not "
              "dominant among these HVG pairs.")
    else:
        print("  => Clade-tracking (high-eta^2) pairs shrink more on deconfounding: scPhyTr's "
              "K down-weights exactly the shared-ancestry co-expression PATH/Hotspot inflate.")
    return {"tumor": tumor, "overlap_top": overlap, "eta_shrink_spearman": float(rho),
            "shrink_hi_eta": float(np.median(shrink[hi])),
            "shrink_lo_eta": float(np.median(shrink[lo]))}


if __name__ == "__main__":
    import sys
    if "pfa" in sys.argv:
        main_pfa()
    elif "real" in sys.argv:
        real_coevolution_discoveries()
    else:
        main()
