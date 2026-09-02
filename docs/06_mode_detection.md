# Hierarchical Bayesian mode detection

Which genes are under selection, and **where on the tree** did that selection change?

The established way to ask the first half of that question is one gene at a time: fit neutral drift,
fit stabilising selection, fit adaptation to a shifted optimum, and pick a winner by an information
criterion. `scPhyTr` has that — [`tl.detect_adaptive`](01_methods.md) — and so does the recent
literature on single-cell lineage trees.

It has two limits that no amount of better fitting removes.

**A hard information floor.** A single gene on a small tree does not carry enough evidence to
separate three hypotheses. SCOUT (Stuart & McKenna 2025) report 0.55 three-way accuracy at 32
leaves, against a 1/3 chance line — and being Bayesian about it does not help, because Bayesian
inference reports uncertainty rather than creating information.

**It cannot tell you where.** Per-gene model selection *tests* an adaptive landscape you supply. It
never asks which branch an event happened on, so it cannot discover one.

`tl.detect_modes` attacks both by sharing structure across genes.

## The model

Adaptive events are latent and live on **branches**, so hundreds of genes each carrying weak
evidence can localise the same event. A second, per-gene indicator decides which genes actually
respond to it — an event is then defined by the genes that respond, not by a label drawn in advance.

```
z_b      ~ Bernoulli(pi)                            which BRANCHES carry an event
g_bg|z_b ~ Bernoulli(z_b * rho)                     which GENES respond to it
d_bg|g   ~ N(0, omega^2 sigma_g^2)  if g_bg else 0  the optimum shift, else exactly 0
theta_b  = theta_parent(b) + d_b                    optima inherited down the tree
Y_g      ~ N(theta0_g 1 + U(alpha) d_g,  sigma_g^2 [R(alpha) + tau I])
```

The tip mean is **linear in the shifts**, so after whitening this is Gaussian linear variable
selection rather than a bespoke phylogenetic optimiser, and every coordinate is conjugate except
`(alpha, tau)`, which sit on a precomputed grid.

## Why it works — three things that are not obvious

**Pooling has to go through the gene indicator.** Summing per-gene evidence over *all* genes ranks
the true branch worse than per-gene testing does, because non-responders swamp responders. The
`z_b` update integrates `gamma` out analytically,

```
logit P(z_b = 1 | .) = logit(pi) + sum_g log[(1 - rho) + rho * BF_bg]
```

so a non-responder has `BF ~ 1`, contributes `~0`, and cannot swamp anything.

**Parent and child branches are exactly collinear on an ultrametric tree.** The design column for a
shift on branch `b` is a *scaled clade indicator*, so a parent's column lies exactly in the span of
any child partition of its clade. The likelihood cannot separate one event at a parent from events
at all its children; only the sparsity prior can, and a sampler that moves one branch at a time
never makes that jump. The sweep therefore updates a parent and its children jointly.

**Do not smooth the data first.** Lineage smoothing multiplies expression by a similarity matrix,
giving a covariance no OU process can produce. Fed to a generative model it is absorbed as signal —
16 spurious events where the truth had one. The `tau` term models the noise instead.

## Using it

```python
ph.pp.setup_anndata(adata, tree)
ph.tl.detect_modes(adata, genes=panel)          # -> adata.uns['modes']

m = adata.uns["modes"]
order = m["evidence"].argsort()[::-1]
m["clades"][order[0]]        # the leaf names below the top-scoring branch
m["p_z"][order[0]]           # posterior P(an event on that branch)
m["p_gamma"][order[0]] > .5  # which genes respond to it
```

Identify a branch by its **clade**, never by its index: `detect_modes` builds its own tree
internally and those indices mean nothing outside the call.

`evidence_only=True` skips the sampler and returns just the graded branch scan. That is much cheaper
and is the more useful read-out at small effect sizes, where `p_z` is sharply thresholded by the
sparsity prior and reports ~0 for everything.

## What it recovers

Against a benchmark where SCOUT is handed the true adaptive landscape and its own preprocessing,
while this is given only the tree and the counts, the branch scan matches or beats an oracle that is
told which genes respond — at every effect size, including ones where per-gene testing is at the
false-positive floor. The responder set comes back at AUROC 0.77–0.91.

## Honest limits

- **It pseudobulks.** The model is Gaussian with one value per tip, so cells are averaged to their
  leaf. On a single-cell tree, where each leaf is one cell, nothing is lost; on a subclone tree the
  within-leaf variation is discarded. The count observation layer is not built for this model yet.
- **`P(z_b = 1 | Y)` is sharply prior-thresholded.** Below a threshold effect size it is ~0 for
  every branch and the ranking it induces is noise. Use `evidence` there.
- **Ultrametric trees**, and a dense factorisation that caps practical size near a few hundred tips
  per call.
- **One event per branch**, no interaction between events.
- **A second tree representation** lives inside `scphytr.modes` — see `modes/_adapt.py` for why.

## Where it came from

Developed as the `sparseou` subproject against SCOUT (bioRxiv 2025.11.12.688020). The comparison
machinery — a validated port of SCOUT's preprocessing, runners for the original R package, the Stan
per-gene Bayesian baseline and its specification — is in `analysis/scout/`, along with the full
development history as a git bundle.
