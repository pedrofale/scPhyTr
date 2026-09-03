"""Recovery study I: does Stan recover the parameters of a gene generated under BM1?

The gene is simulated from exactly the model being fitted, and BM1 is fitted ALONE
(``fit_model=1``) -- no model comparison, no competing hypotheses. Data are NOT standardised, so
sigma, tau and theta keep their generating units and recovery is a real check rather than a check
up to an unknown rescaling.

The interesting question for BM is not "is sigma recovered" but "which parameters are recoverable
at all". Every tip shares one root, so the root state theta has an analytic posterior sd

    sd(theta) = ( 1' V^-1 1 )^{-1/2},      V = sigma^2 C + tau^2 I

which is far larger than the iid intuition sqrt(var/n) and barely improves with n. That number is
computed before sampling and is the sharpest available test of the implementation.

    python -m analysis.scout.experiments.08_recovery_bm
"""
import os, time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import arviz as az

from scphytr.modes._tree import Tree
from scphytr.modes._ou import tip_cov
from scphytr.modes.simulate import simulate_tips, leaf_regimes
from analysis.scout.stan_scout import get_model, fit_gene

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "results", "recovery_bm")
SIGMA_T, TAU_T, THETA_T = 1.0, 0.3, 2.0          # truth
PRIORS = dict(use_tau=True, s_theta=5.0, s_scale=3.0, s_tau=1.0, sd_log_hl=1.0,
              standardize=False, fit_model=1)     # BM1 alone, raw units


def analytic_sd_theta(tree, sigma=SIGMA_T, tau=TAU_T):
    C = tip_cov(tree, 0.0, 1.0, root="fixed")
    V = sigma ** 2 * C + tau ** 2 * np.eye(tree.n_leaves)
    one = np.ones(tree.n_leaves)
    return float(1.0 / float(one @ np.linalg.solve(V, one))) ** 0.5


def simulate(tree, seed):
    d0 = np.zeros((tree.n_nodes, 1))
    return simulate_tips(tree, 0.0, SIGMA_T, np.array([THETA_T]), d0,
                         rng=np.random.default_rng(seed), root="fixed", sigma_obs=TAU_T)[:, 0]


def main():
    model = get_model()
    rows, idata_128 = [], None
    print("truth:  theta = %.1f   sigma = %.1f   tau = %.1f   (BM1 fitted alone, raw units)\n"
          % (THETA_T, SIGMA_T, TAU_T))
    print(f"{'n':>5} {'theta':>16} {'sigma':>16} {'tau':>16} {'sd(theta) pred':>15} {'rhat':>6} {'div':>4}")

    for n in (32, 64, 128, 256):
        tree = Tree.balanced(n, 1.0).scale_height(1.0)
        z = np.zeros(tree.n_nodes, bool)
        z[next(v for v in range(1, tree.n_nodes) if len(tree.leaf_sets()[v]) == n // 4)] = True
        lr = leaf_regimes(tree, z)                       # unused by BM1, but the data block needs it
        y = simulate(tree, seed=100 + n)
        pred = analytic_sd_theta(tree)
        t0 = time.time()
        s, fit = fit_gene(tree, y, lr, model=model, chains=4, iter_warmup=1000,
                          iter_sampling=1000, seed=7, **PRIORS)
        idata = az.from_cmdstanpy(posterior=fit)
        if n == 128:
            idata_128 = idata
        post = idata.posterior
        th = post["theta_bm"].values.ravel(); sg = post["sigma_bm"].values.ravel()
        ta = post["tau"].values.reshape(-1, 3)[:, 0]
        sm = az.summary(idata, var_names=["theta_bm", "sigma_bm", "tau"], round_to=4)
        rh = float(sm.loc[["theta_bm", "sigma_bm", "tau[0]"], "r_hat"].max())
        dv = int(fit.method_variables()["divergent__"].sum())
        rows.append(dict(n=n, theta_mean=th.mean(), theta_sd=th.std(), theta_pred_sd=pred,
                         sigma_mean=sg.mean(), sigma_sd=sg.std(),
                         tau_mean=ta.mean(), tau_sd=ta.std(), rhat=rh, div=dv,
                         secs=time.time() - t0))
        r = rows[-1]
        print(f"{n:5d} {r['theta_mean']:8.3f}+-{r['theta_sd']:.3f} "
              f"{r['sigma_mean']:8.3f}+-{r['sigma_sd']:.3f} {r['tau_mean']:8.3f}+-{r['tau_sd']:.3f} "
              f"{pred:15.3f} {rh:6.3f} {dv:4d}")

    df = pd.DataFrame(rows); df.to_csv(f"{OUT}/recovery.csv", index=False)
    with open(f"{OUT}/summary.txt", "w") as fh:
        fh.write("truth: theta=%.1f sigma=%.1f tau=%.1f\n\n" % (THETA_T, SIGMA_T, TAU_T))
        fh.write(df.to_string(index=False) + "\n\n")
        fh.write(az.summary(idata_128, var_names=["theta_bm", "sigma_bm", "tau"],
                            round_to=4).to_string() + "\n")
    print("\n" + az.summary(idata_128, var_names=["theta_bm", "sigma_bm", "tau"],
                            round_to=4).to_string())

    # ---- 1. posteriors vs truth (n = 128) ----
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
    for ax, (v, truth, lab) in zip(axes, [("theta_bm", THETA_T, r"$\theta_{bm}$  (root state)"),
                                          ("sigma_bm", SIGMA_T, r"$\sigma_{bm}$  (drift rate)"),
                                          ("tau", TAU_T, r"$\tau$  (measurement error)")]):
        d = (idata_128.posterior[v].values.reshape(-1, 3)[:, 0] if v == "tau"
             else idata_128.posterior[v].values.ravel())
        ax.hist(d, bins=60, color="#6B7A99", alpha=.85, edgecolor="none")
        ax.axvline(truth, color="#B5762A", lw=2.2, ls="--")
        ax.set_title(lab, fontsize=11); ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
    plt.suptitle("BM1 fitted to a BM1 gene, 128 tips — dashed = generating value", y=1.03)
    fig.tight_layout(); fig.savefig(f"{OUT}/1_posteriors.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    # ---- 2. the root state does not sharpen with n ----
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.plot(df.n, df.theta_sd, "o-", color="#6B7A99", lw=2, label="posterior sd (Stan)")
    ax.plot(df.n, df.theta_pred_sd, "s--", color="#171A21", lw=1.6, label=r"analytic $(1'V^{-1}1)^{-1/2}$")
    ax.plot(df.n, np.sqrt((SIGMA_T**2 + TAU_T**2) / df.n), "^:", color="#B5762A", lw=1.6,
            label=r"if tips were iid:  $\sqrt{\mathrm{var}/n}$")
    ax.set_xscale("log", base=2); ax.set_xticks(df.n); ax.set_xticklabels(df.n)
    ax.set_xlabel("number of tips"); ax.set_ylabel(r"sd of $\theta_{bm}$")
    ax.set_title("The root state is the parameter phylogeny will not give you")
    ax.legend(frameon=False, fontsize=9); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/2_theta_vs_n.png", dpi=140); plt.close(fig)

    # ---- 3. diagnostics ----
    az.plot_trace(idata_128, var_names=["theta_bm", "sigma_bm"], figsize=(11, 4.5))
    plt.suptitle("Traces, BM1 gene at 128 tips", y=1.01)
    plt.tight_layout(); plt.savefig(f"{OUT}/3_trace.png", dpi=140, bbox_inches="tight"); plt.close()

    az.plot_pair(idata_128, var_names=["theta_bm", "sigma_bm"], divergences=True,
                 figsize=(5.2, 5), scatter_kwargs={"alpha": .25, "s": 8},
                 marginals=True, reference_values={"theta_bm": THETA_T, "sigma_bm": SIGMA_T},
                 reference_values_kwargs={"color": "#B5762A", "ms": 11, "marker": "*"})
    plt.suptitle("Joint posterior; star = truth", y=1.01)
    plt.tight_layout(); plt.savefig(f"{OUT}/4_pair.png", dpi=140, bbox_inches="tight"); plt.close()

    idata_128.to_netcdf(f"{OUT}/idata_bm_128.nc")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
