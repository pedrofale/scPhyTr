"""Felsenstein (1985) Fig. 5-7, in gene/factor space.

Felsenstein's "worst case": 40 species in two clades of 20 close relatives.
Two characters evolve by *independent* Brownian motion (no true relationship),
yet a phylogeny-naive regression finds an "illusory" significant correlation,
because there are really ~2-3 independent points, not 40.

Here we reproduce that argument for *factor analysis* on transcriptome-like
data. The genes are evolutionarily **independent** (the evolutionary rate
matrix K is diagonal), so there is **no true low-rank gene program / factor**.
We then ask whether a factor is "significant" using Horn's parallel analysis,
run two ways with the *same* test:

  * naive  -- on the raw leaf values, treating the n cells as i.i.d.;
  * phylo  -- on Felsenstein's independent contrasts (Cholesky-whitened by the
              tree covariance C), which restores independence.

The naive test routinely declares a spurious top factor whose loadings are just
the deep clade-discriminating axis and whose score is "which clade are you in".
The contrast (phylo) test, the multivariate analogue of analysing Felsenstein's
contrasts, correctly finds nothing. A repeated-histories panel quantifies the
false-positive rate (his "illusory P < .05").

Outputs docs/figures/felsenstein.png and prints the numbers used in the
write-up. Run inside the `scphytr` conda env. Deterministic.
"""

import os
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mplcache"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scphytr.utils.tree import Tree
from scphytr.utils.covariance import bm_covariance

HERE = os.path.dirname(__file__)


# ---------------------------------------------------------------------------
# Felsenstein's "worst case" tree: two clades of close relatives.
# ---------------------------------------------------------------------------

def two_clade_newick(n_per_clade=20, deep=9.0, tip=1.0):
    """Two clades, each a star of ``n_per_clade`` tips on short terminal branches.

    Ultrametric: root-to-tip = ``deep + tip``. Within-clade pairs share ``deep``
    of their history; between-clade pairs share nothing (they split at the root).
    With deep/tip = 9 the within-clade correlation is 0.9 -- strong pseudo-
    replication, so the effective number of independent points is ~2-3.
    """
    def clade(prefix):
        tips = ",".join(f"{prefix}{i}:{tip}" for i in range(n_per_clade))
        return f"({tips}):{deep}"
    return f"({clade('A')},{clade('B')});"


# ---------------------------------------------------------------------------
# Independent genes on the tree: K diagonal -> NO true factor.
# ---------------------------------------------------------------------------

def simulate_independent_genes(L_C, p, noise_sd, rng):
    """Each gene is an *independent* Brownian motion on the tree (K = diagonal).

    L_C is the Cholesky factor of the tree covariance C (n x n). Returns Y (n, p).
    There is, by construction, no shared factor: any factor a method 'finds' is
    an artefact of the phylogeny.

    As in Felsenstein (1985), the character is observed directly (``noise_sd=0``):
    the data follow BM exactly, so whitening by ``C`` yields exactly i.i.d.
    contrasts. With measurement noise the trait covariance is ``C + Psi`` and the
    correct whitening is by that marginal -- which is precisely what phylogenetic
    factor analysis does through its idiosyncratic-variance term ``Psi``.
    """
    n = L_C.shape[0]
    Z = L_C @ rng.standard_normal((n, p))           # (n, p) independent BM genes
    if noise_sd:
        Z = Z + noise_sd * rng.standard_normal((n, p))
    return Z


# ---------------------------------------------------------------------------
# Felsenstein contrasts via Cholesky whitening (with GLS phylogenetic mean).
# ---------------------------------------------------------------------------

def phylo_contrasts(Y, C):
    """Felsenstein's n-1 standardized independent contrasts (multivariate).

    Build an orthonormal basis ``H`` (n x n-1) of the contrast space (the
    orthogonal complement of the all-ones vector, so ``H^T 1 = 0``): the columns
    of ``H^T Y`` are ordinary contrasts with the unknown root/mean removed. Their
    per-gene covariance is ``H^T C H``; whitening by its Cholesky factor ``G``
    gives ``G^{-1} H^T Y``, whose n-1 rows are i.i.d. ``N(0, sigma_g^2 I)`` for an
    independent-BM gene. Unlike a plain Cholesky whitening of ``C`` (which leaves
    a constraint against the non-constant vector ``L^{-1} 1`` and thus
    non-exchangeable rows), these contrasts are genuinely exchangeable, so the
    permutation null of parallel analysis is valid on them.
    """
    n = C.shape[0]
    M = np.eye(n) - np.ones((n, n)) / n
    Uc, _, _ = np.linalg.svd(M)
    H = Uc[:, : n - 1]                              # (n, n-1), H^T 1 = 0, orthonormal
    Hc = H.T @ C @ H                                # (n-1, n-1)
    G = np.linalg.cholesky(Hc)
    return np.linalg.solve(G, H.T @ Y)             # (n-1, p), i.i.d. rows


# ---------------------------------------------------------------------------
# Horn's parallel analysis: is the top eigenvalue above the i.i.d. null?
# ---------------------------------------------------------------------------

def parallel_analysis(M, n_perm, rng):
    """Number of 'significant' factors by Horn's parallel analysis.

    Treats the rows of ``M`` as exchangeable samples. The null shuffles each
    column (gene) independently across rows, destroying any cross-gene structure
    while preserving the marginal of each gene. A factor is significant if its
    sample-covariance eigenvalue exceeds the 95th percentile of the null. The
    test is *valid only if the rows are genuinely i.i.d.* -- true for contrasts,
    false for raw phylogenetic leaves.
    """
    m, p = M.shape
    Mc = M - M.mean(0)
    obs = np.sort(np.linalg.eigvalsh((Mc.T @ Mc) / (m - 1)))[::-1]
    null = np.empty((n_perm, p))
    for b in range(n_perm):
        P = np.empty_like(Mc)
        for g in range(p):
            P[:, g] = Mc[rng.permutation(m), g]
        null[b] = np.sort(np.linalg.eigvalsh((P.T @ P) / (m - 1)))[::-1]
    null95 = np.percentile(null, 95, axis=0)
    # Horn's rule: retain leading consecutive components that beat the null,
    # stopping at the first that does not (counting *any* component would inflate
    # the false-positive rate by multiple comparisons across the p components).
    n_sig = 0
    for i in range(p):
        if obs[i] > null95[i]:
            n_sig += 1
        else:
            break
    return obs, null95, n_sig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    n_per_clade, p = 20, 12
    n = 2 * n_per_clade
    tree = Tree(two_clade_newick(n_per_clade, deep=9.0, tip=1.0))
    leaf_names = [l.name for l in tree.root.get_leaves()]
    clade = np.array([0 if name.startswith("A") else 1 for name in leaf_names])

    C = bm_covariance(tree)
    L_C = np.linalg.cholesky(C + 1e-10 * np.eye(n))

    # ----- one representative dataset (Felsenstein's Fig. 6/7) ---------------
    rng = np.random.default_rng(1)
    Y = simulate_independent_genes(L_C, p, noise_sd=0.0, rng=rng)

    obs_naive, null_naive, sig_naive = parallel_analysis(Y, n_perm=500, rng=np.random.default_rng(10))
    Yc = phylo_contrasts(Y, C)
    obs_phy, null_phy, sig_phy = parallel_analysis(Yc, n_perm=500, rng=np.random.default_rng(11))

    # naive PCA for the scatter / score panels
    Ycen = Y - Y.mean(0)
    Up, sp, Vt = np.linalg.svd(Ycen, full_matrices=False)
    pc = Up[:, :2] * sp[:2]                          # (n, 2) PC scores
    var_explained = sp ** 2 / np.sum(sp ** 2)
    pc1_loading = Vt[0]                              # (p,)
    clade_axis = Y[clade == 0].mean(0) - Y[clade == 1].mean(0)
    clade_axis /= np.linalg.norm(clade_axis)
    cos_pc1_clade = abs(pc1_loading @ clade_axis)
    corr_pc1_clade = abs(np.corrcoef(pc[:, 0], clade)[0, 1])

    print("\n=== Felsenstein worst case, in factor space ===")
    print(f"tree: {n} leaves in 2 clades of {n_per_clade}; p={p} genes; "
          f"within-clade corr = {C[0,1]/C[0,0]:.2f}, between-clade = {C[0,n-1]/C[0,0]:.2f}")
    print("genes are EVOLUTIONARILY INDEPENDENT -> there is no true factor.")
    print(f"naive parallel analysis: {sig_naive} 'significant' factor(s); "
          f"PC1 explains {var_explained[0]*100:.0f}% of variance")
    print(f"   naive PC1 vs clade: |corr(score, clade)| = {corr_pc1_clade:.3f}, "
          f"|cos(loading, clade axis)| = {cos_pc1_clade:.3f}  "
          f"(the 'factor' is just the deep split)")
    print(f"phylo (contrast) parallel analysis: {sig_phy} significant factor(s) "
          f"-- correctly none")

    # ----- repeated histories: false-positive rate --------------------------
    R = 300
    fp_naive = np.zeros(R, dtype=int)
    fp_phy = np.zeros(R, dtype=int)
    rng2 = np.random.default_rng(2025)
    for r in range(R):
        Yr = simulate_independent_genes(L_C, p, noise_sd=0.0, rng=rng2)
        _, _, sN = parallel_analysis(Yr, n_perm=200, rng=np.random.default_rng(r))
        Ycr = phylo_contrasts(Yr, C)
        _, _, sP = parallel_analysis(Ycr, n_perm=200, rng=np.random.default_rng(1000 + r))
        fp_naive[r] = sN
        fp_phy[r] = sP
    rate_naive = float(np.mean(fp_naive > 0))
    rate_phy = float(np.mean(fp_phy > 0))
    print(f"\nover {R} independent evolutionary histories (genes always independent):")
    print(f"   P(naive finds >=1 factor)   = {rate_naive:.2f}   <- illusory factors")
    print(f"   P(phylo finds >=1 factor)   = {rate_phy:.2f}   <- ~nominal 0.05")
    print(f"   mean #factors  naive={fp_naive.mean():.2f}  phylo={fp_phy.mean():.2f}")

    # ----- figure -----------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.8))

    # (A) naive PC space: two clusters = the deep split (Felsenstein Fig. 7)
    for cl, col, lab in [(0, "#2c7fb8", "clade A"), (1, "#d95f0e", "clade B")]:
        m = clade == cl
        ax[0, 0].scatter(pc[m, 0], pc[m, 1], s=28, alpha=0.8, color=col, label=lab)
    ax[0, 0].set_xlabel(f"naive PC1 ({var_explained[0]*100:.0f}% var)")
    ax[0, 0].set_ylabel(f"naive PC2 ({var_explained[1]*100:.0f}% var)")
    ax[0, 0].legend(fontsize=9)
    ax[0, 0].set_title("(A) Naive factor analysis of independent genes\n"
                       "PC1 separates the two clades", fontsize=11)

    # (B) parallel analysis: same test, raw tips vs contrasts
    idx = np.arange(1, p + 1)
    ax[0, 1].plot(idx, obs_naive, "o-", color="#d95f0e", label="naive: observed")
    ax[0, 1].plot(idx, null_naive, "s--", color="#d95f0e", alpha=0.45,
                  label="naive: null (95%)")
    ax[0, 1].plot(idx, obs_phy, "o-", color="#2c7fb8", label="phylo (contrasts): observed")
    ax[0, 1].plot(idx, null_phy, "s--", color="#2c7fb8", alpha=0.45,
                  label="phylo: null (95%)")
    ax[0, 1].set_yscale("log")
    ax[0, 1].set_xlabel("component")
    ax[0, 1].set_ylabel("eigenvalue (log scale)")
    ax[0, 1].legend(fontsize=8)
    ax[0, 1].set_title("(B) Horn's parallel analysis\n"
                       "naive top eigenvalue pokes above its null; contrasts do not",
                       fontsize=11)
    ax[0, 1].annotate("illusory\nfactor", xy=(1, obs_naive[0]),
                      xytext=(2.4, obs_naive[0] * 1.1), fontsize=9, color="#d95f0e",
                      arrowprops=dict(arrowstyle="->", color="#d95f0e"))

    # (C) the naive 'factor' IS the phylogeny: PC1 score by clade
    ax[1, 0].hist(pc[clade == 0, 0], bins=12, alpha=0.7, color="#2c7fb8", label="clade A")
    ax[1, 0].hist(pc[clade == 1, 0], bins=12, alpha=0.7, color="#d95f0e", label="clade B")
    ax[1, 0].set_xlabel("naive PC1 score")
    ax[1, 0].set_ylabel("# cells")
    ax[1, 0].legend(fontsize=9)
    ax[1, 0].set_title(f"(C) The 'factor' is clade membership\n"
                       f"|corr(PC1, clade)| = {corr_pc1_clade:.2f}, "
                       f"|cos(loading, clade axis)| = {cos_pc1_clade:.2f}", fontsize=11)

    # (D) repeated histories: illusory-significance false-positive rate
    bars = ax[1, 1].bar([0, 1], [rate_naive, rate_phy],
                        color=["#d95f0e", "#2c7fb8"], width=0.6)
    ax[1, 1].axhline(0.05, ls=":", color="grey", lw=1.2)
    ax[1, 1].text(1.45, 0.07, "nominal 0.05", fontsize=9, color="grey", ha="right")
    ax[1, 1].set_xticks([0, 1])
    ax[1, 1].set_xticklabels(["naive", "phylo\n(contrasts)"])
    ax[1, 1].set_ylabel(f"P(\u22651 'significant' factor)  over {R} histories")
    ax[1, 1].set_ylim(0, 1.05)
    for b, v in zip(bars, [rate_naive, rate_phy]):
        ax[1, 1].text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                      ha="center", fontsize=10)
    ax[1, 1].set_title("(D) Illusory significance (Felsenstein's P < .05)\n"
                       "genes are independent in every history", fontsize=11)

    fig.tight_layout()
    out = os.path.join(HERE, "felsenstein.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\n[fig] wrote {out}")


if __name__ == "__main__":
    main()
