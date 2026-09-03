"""Frequentist coverage at fixed truth: does the 90% credible interval contain it 90% of the time?"""
import sys, time; sys.path.insert(0,'src')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scphytr.modes._tree import Tree
from scphytr.modes.simulate import simulate_tips, leaf_regimes
from analysis.scout.stan_scout import get_model, fit_gene

SIGMA_T, TAU_T, THETA_T, N, R = 1.0, 0.3, 2.0, 128, 150
model = get_model()
tree = Tree.balanced(N, 1.0).scale_height(1.0)
z = np.zeros(tree.n_nodes, bool)
z[next(v for v in range(1,tree.n_nodes) if len(tree.leaf_sets()[v])==N//4)] = True
lr = leaf_regimes(tree, z)
d0 = np.zeros((tree.n_nodes,1))
P = dict(use_tau=True, s_theta=5.0, s_scale=3.0, s_tau=1.0, sd_log_hl=1.0,
         standardize=False, fit_model=1)

rows=[]; t0=time.time()
for r in range(R):
    y = simulate_tips(tree, 0.0, SIGMA_T, np.array([THETA_T]), d0,
                      rng=np.random.default_rng(9000+r), root="fixed", sigma_obs=TAU_T)[:,0]
    s, fit = fit_gene(tree, y, lr, model=model, chains=2, iter_warmup=700,
                      iter_sampling=700, seed=r+1, **P)
    d = fit.stan_variables()
    rec = dict(rep=r, div=int(fit.method_variables()["divergent__"].sum()))
    for nm, draws, truth in (("theta", d["theta_bm"], THETA_T),
                             ("sigma", d["sigma_bm"], SIGMA_T),
                             ("tau",   d["tau"][:,0], TAU_T)):
        dr = np.asarray(draws).ravel()
        for lvl in (50, 90):
            lo, hi = np.percentile(dr, [(100-lvl)/2, 100-(100-lvl)/2])
            rec[f"{nm}_cov{lvl}"] = int(lo <= truth <= hi)
        rec[f"{nm}_mean"] = dr.mean(); rec[f"{nm}_sd"] = dr.std()
        rec[f"{nm}_rank"] = float((dr < truth).mean())      # posterior prob below truth
    rows.append(rec)
    if (r+1) % 30 == 0: print(f"  {r+1}/{R}  ({time.time()-t0:.0f}s)", flush=True)

df = pd.DataFrame(rows); df.to_csv("results/recovery_bm/coverage.csv", index=False)
se = lambda p,n: 1.96*np.sqrt(p*(1-p)/n)
print(f"\nCoverage over {R} independent datasets, n={N} tips  (truth theta={THETA_T} sigma={SIGMA_T} tau={TAU_T})")
print(f"{'param':>7} {'50% CI':>18} {'90% CI':>18}   {'mean(post.mean)':>16}")
for nm, truth in (("theta",THETA_T),("sigma",SIGMA_T),("tau",TAU_T)):
    c50, c90 = df[f"{nm}_cov50"].mean(), df[f"{nm}_cov90"].mean()
    print(f"{nm:>7} {c50:8.3f} +-{se(c50,R):5.3f} {c90:8.3f} +-{se(c90,R):5.3f}   "
          f"{df[f'{nm}_mean'].mean():8.3f}  (truth {truth})")
print(f"total divergences across all {R} fits: {df['div'].sum()}")

fig, axes = plt.subplots(1,3, figsize=(12.5,3.5))
for ax,(nm,lab) in zip(axes, [("theta",r"$\theta_{bm}$"),("sigma",r"$\sigma_{bm}$"),("tau",r"$\tau$")]):
    ax.hist(df[f"{nm}_rank"], bins=np.linspace(0,1,16), color="#6B7A99", alpha=.85)
    ax.axhline(R/15, color="#B5762A", lw=2, ls="--")
    ax.set_title(f"{lab}: posterior prob. below truth", fontsize=10)
    ax.set_xlabel("should be Uniform(0,1)"); ax.set_yticks([])
    ax.spines[["top","right","left"]].set_visible(False)
plt.suptitle(f"Calibration over {R} independent BM datasets — flat is correct", y=1.03)
plt.tight_layout(); plt.savefig("results/recovery_bm/5_calibration.png", dpi=140, bbox_inches="tight"); plt.close()

fig, ax = plt.subplots(figsize=(5,4.4))
ax.scatter(df.sigma_mean, df.tau_mean, s=18, alpha=.6, color="#6B7A99")
ax.scatter([SIGMA_T],[TAU_T], marker="*", s=220, color="#B5762A", zorder=5, label="truth")
ax.set_xlabel(r"posterior mean $\sigma_{bm}$"); ax.set_ylabel(r"posterior mean $\tau$")
r = np.corrcoef(df.sigma_mean, df.tau_mean)[0,1]
ax.set_title(f"drift vs noise trade off across datasets (r = {r:.2f})", fontsize=10)
ax.legend(frameon=False); ax.spines[["top","right"]].set_visible(False)
plt.tight_layout(); plt.savefig("results/recovery_bm/6_sigma_tau.png", dpi=140); plt.close()
print("wrote calibration + sigma/tau figures")
