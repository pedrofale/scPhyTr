"""Run the Stan model on genes with KNOWN truth and inspect the posteriors with ArviZ.

Three genes are simulated on one 128-tip tree with a single regime split, one from each hypothesis
class, so every posterior can be read against the value that generated it:

    BM1  neutral drift                     alpha = 0
    OU1  one global optimum                alpha = ALPHA_TRUE, no optimum shift
    OUx  one optimum per regime            alpha = ALPHA_TRUE, optimum shifts by SHIFT

alpha is invariant to affine rescaling of y, so it is directly comparable to truth even though the
model standardises its input; optimum contrasts are rescaled by sd(y) before comparison.

    PYTHONPATH=src python experiments/07_stan_posteriors.py
"""
import os, sys, time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import arviz as az

from sparseou.tree import Tree
from sparseou.simulate import simulate_tips, leaf_regimes
from sparseou.stan_scout import get_model, fit_gene, stan_data

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "results", "arviz")
N_TIPS, ALPHA_TRUE, S_TRUE, SHIFT = 128, 1.5, 1.0, 1.5
SIGMA_TRUE = S_TRUE * np.sqrt(2 * ALPHA_TRUE)          # stationary sd -> BM-style sigma


def build():
    tree = Tree.balanced(N_TIPS, 1.0).scale_height(1.0)
    sets = tree.leaf_sets()
    b = int(next(v for v in range(1, tree.n_nodes) if len(sets[v]) == N_TIPS // 4))
    z = np.zeros(tree.n_nodes, bool); z[b] = True
    lr = leaf_regimes(tree, z)
    d0 = np.zeros((tree.n_nodes, 1))
    d1 = np.zeros((tree.n_nodes, 1)); d1[b, 0] = SHIFT
    genes = {
        "BM1": simulate_tips(tree, 1e-8, SIGMA_TRUE, np.array([0.0]), d0,
                             rng=np.random.default_rng(11), root="fixed")[:, 0],
        "OU1": simulate_tips(tree, ALPHA_TRUE, SIGMA_TRUE, np.array([0.0]), d0,
                             rng=np.random.default_rng(12), root="stationary")[:, 0],
        "OUx": simulate_tips(tree, ALPHA_TRUE, SIGMA_TRUE, np.array([0.0]), d1,
                             rng=np.random.default_rng(13), root="stationary")[:, 0],
    }
    return tree, lr, b, genes


def main():
    tree, lr, b, genes = build()
    model = get_model()
    print(f"{N_TIPS} tips, regime split at node {b} (|clade|={len(tree.leaf_sets()[b])}), "
          f"true alpha={ALPHA_TRUE}, true optimum shift={SHIFT}\n")

    idatas, summaries = {}, {}
    for name, y in genes.items():
        t0 = time.time()
        s, fit = fit_gene(tree, y, lr, model=model, chains=4, iter_warmup=1000,
                          iter_sampling=1000, seed=20, use_tau=True,
                          s_theta=1.0, s_scale=1.0, sd_log_hl=1.0)
        idatas[name] = az.from_cmdstanpy(posterior=fit)
        summaries[name] = s
        sd = float(np.std(y))
        print(f"--- truth = {name}  [{time.time()-t0:.0f}s] "
              f"P(BM1)={s['p_BM1']:.3f}  P(OU1)={s['p_OU1']:.3f}  P(OUx)={s['p_OUX']:.3f} "
              f" -> call {s['call']}   [sd(y)={sd:.2f}]")

    # ---------- printed diagnostics ----------
    VARS = ["p_model", "alpha_ou1", "alpha_oux", "theta_oux", "s_oux", "sigma_bm",
            "theta_spread", "tau"]
    with open(os.path.join(OUT, "summaries.txt"), "w") as fh:
        for name, idata in idatas.items():
            tbl = az.summary(idata, var_names=VARS, round_to=3)
            hdr = f"\n{'='*78}\nTRUTH = {name}\n{'='*78}"
            print(hdr); print(tbl.to_string())
            fh.write(hdr + "\n" + tbl.to_string() + "\n")

    # ---------- 1. model probabilities vs truth ----------
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    labs, x, w = ["BM1", "OU1", "OUX"], np.arange(3), 0.26
    cols = ["#6B7A99", "#2F7D74", "#B5762A"]
    for i, (name, s) in enumerate(summaries.items()):
        ax.bar(x + (i - 1) * w, [s["p_BM1"], s["p_OU1"], s["p_OUX"]], w,
               label=f"truth {name}", color=cols[i], edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(["P(BM1|y)", "P(OU1|y)", "P(OUx|y)"])
    ax.set_ylim(0, 1.05); ax.set_ylabel("posterior probability")
    ax.set_title("Posterior model probability, by the hypothesis that generated the gene")
    ax.legend(frameon=False, fontsize=9); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/1_model_probabilities.png", dpi=140); plt.close(fig)

    # ---------- 2. trace, OUx gene ----------
    az.plot_trace(idatas["OUx"], var_names=["alpha_oux", "s_oux", "theta_oux", "theta_spread"],
                  figsize=(11, 8))
    plt.suptitle("Traces — gene generated under OUx", y=1.005)
    plt.tight_layout(); plt.savefig(f"{OUT}/2_trace_oux.png", dpi=140, bbox_inches="tight"); plt.close()

    # ---------- 3. the alpha-scale ridge, with divergences ----------
    az.plot_pair(idatas["OUx"], var_names=["hl_oux", "s_oux", "theta_spread"],
                 divergences=True, figsize=(8.5, 8), scatter_kwargs={"alpha": .25, "s": 8})
    plt.suptitle("Joint geometry — half-life vs scale vs optimum spread (OUx gene)", y=1.002)
    plt.tight_layout(); plt.savefig(f"{OUT}/3_pair_ridge.png", dpi=140, bbox_inches="tight"); plt.close()

    # ---------- 4. energy + rank ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    az.plot_energy(idatas["OUx"], ax=axes[0])
    axes[0].set_title("Energy (OUx gene)")
    az.plot_rank(idatas["OUx"], var_names=["alpha_oux"], ax=axes[1])
    fig.tight_layout(); fig.savefig(f"{OUT}/4_energy_rank.png", dpi=140); plt.close(fig)

    # ---------- 5. recovery of alpha against truth ----------
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharex=True)
    for ax, (name, idata) in zip(axes, idatas.items()):
        az.plot_posterior(idata, var_names=["alpha_oux"], ax=ax, hdi_prob=.94,
                          ref_val=0.0 if name == "BM1" else ALPHA_TRUE)
        ax.set_title(f"truth {name}" + ("  (alpha = 0)" if name == "BM1"
                                        else f"  (alpha = {ALPHA_TRUE})"))
        ax.set_xlim(0, 8)
    plt.suptitle("Posterior for selection strength under the OUx fit; line = truth", y=1.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/5_alpha_recovery.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ---------- 6. optima, rescaled to standardised units ----------
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
    for ax, (name, idata) in zip(axes, idatas.items()):
        th = idata.posterior["theta_oux"].values.reshape(-1, 2)
        contrast = th[:, 1] - th[:, 0]
        true_c = (SHIFT if name == "OUx" else 0.0) / float(np.std(genes[name]))
        ax.hist(contrast, bins=60, color="#B5762A", alpha=.8, edgecolor="none")
        ax.axvline(true_c, color="#171A21", lw=2, ls="--")
        ax.axvline(0, color="#868D9C", lw=1)
        ax.set_title(f"truth {name}   (true contrast {true_c:+.2f})", fontsize=10)
        ax.set_xlabel(r"$\theta_2 - \theta_1$  (standardised)")
        ax.spines[["top", "right"]].set_visible(False)
    plt.suptitle("Posterior for the optimum contrast; dashed = truth", y=1.03)
    fig.tight_layout(); fig.savefig(f"{OUT}/6_optimum_contrast.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    for k, v in idatas.items():
        v.to_netcdf(f"{OUT}/idata_{k}.nc")
    print(f"\nwrote figures + summaries + netcdf to {OUT}")


if __name__ == "__main__":
    main()
