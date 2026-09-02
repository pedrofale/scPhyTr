# SCOUT digest — what it does, and the gap we attack

**Paper.** Stuart H & McKenna A. *SCOUT: Ornstein–Uhlenbeck modelling of gene expression evolution
on single-cell lineage trees.* bioRxiv 2025.11.12.688020 (posted 2025-11-13). Dartmouth.
Code: <https://github.com/hrstuart/SCOUT> (R package, MIT).

---

## 1. What SCOUT actually is

Given (a) a single-cell lineage tree in Newick, (b) a cell x gene matrix, and (c) **a user-supplied
partition of the leaves into "regimes"**, SCOUT fits, **independently for every gene**, three
hypotheses and picks a winner:

| model | meaning |
|---|---|
| **BM1** | neutral drift (Brownian motion) |
| **OU1** | constrained: one global optimum theta |
| **OUx** | adaptive: one optimum per regime (x = number of regimes) |

Mechanics, precisely:

- The OU fits are **OUwie** (Beaulieu et al. 2012) with the `three.point` algorithm (Ho & Ane 2014)
  for the likelihood; per gene, per model, default parameters.
- **Regimes on internal branches are not inferred from expression.** Leaf regime labels are given by
  the user; internal nodes are painted by **ancestral character estimation** (`ape::ace`, equal-rates
  Mk model, max-likelihood state). Requires a fully dichotomous tree.
- **Model selection = minimum AICc.** Optional stringency: keep only genes with dAICc > 2 to the next
  model; on real data they additionally discard OU fits with alpha < 0.1 as indistinguishable from BM.
- Branch lengths default to 1 when absent.
- Three preprocessing/inference flavours: **SCOUT-SM** (lineage smoothing then OU), **SCOUT-EM**
  (latent expression state, EM: E-step imputes latent expression, M-step fits OU), **MTF** (tip-fog
  measurement-error model, Beaulieu & O'Meara 2025).
- **Lineage smoothing** is their headline noise fix: a Gaussian kernel on *lineage* distance, width
  set by the k-th nearest lineage neighbour; smoothed expression = (lineage similarity matrix) x
  (log-normalised counts). One parameter `k` (default 8).

## 2. Their benchmark (this is what we reproduce)

Simulation pipeline, three stages:

1. **TedSim** simulates the lineage tree, per-cell states, and an indel character matrix.
   Cell-state tree `((t3:1,t4:1):1);` -> **3 states**; p(asymmetric division) = 0.4, step = 0.4,
   N_char = 64.
2. **OUwie.sim** generates, per gene, a continuous OU trait on that tree given (alpha, sigma^2, theta)
   and the regime annotation. *This continuous vector is their "ground truth" arm.*
3. **SymSim** turns the OU trait into realistic counts: the OU trait replaces SymSim's multivariate-
   normal "extrinsic variation" variable; counts follow a Beta-Poisson kinetic model,
   `p ~ Beta(k_on, k_off)`, `X ~ Poisson(p * s)`.

**Headline figure (Fig 2B).** 3-way classification accuracy (caret) of BM1 / OU1 / OU3:
- tree sizes **32 -> 4096** leaves (powers of two),
- **50 genes per model class** (150 genes total),
- **alpha = 0.75, sigma = 0.75, k = 8**,
- three arms: ground-truth continuous trait / log-normalised counts / log-norm + smoothing.

Reported numbers we can target: at **32 leaves** accuracy is **0.55 [0.47, 0.62]** (log-norm) and
**0.52 [0.44, 0.59]** (smoothed) — i.e. barely above the 1/3 chance line. Smoothing "approaches
ground truth for trees as small as 128 leaves"; log-normalisation alone stays consistently low.
Fig 2C sweeps `k` at 512 cells; Fig 2D is an alpha x sigma grid — **worst when sigma is high and
alpha is low** (OU -> BM degeneracy).

Real data: C. elegans (Packer et al., via `moscot`) and a lung adenocarcinoma xenograft.

## 3. The gap — why a hierarchical sparse OU model is a real contribution

Three structural limitations, all visible in the paper itself:

1. **Genes are fitted independently.** Every gene must carry its own evidence for selection. Their
   own Fig 2B shows this failing at small trees (accuracy 0.55 at 32 leaves) — exactly the
   weak-per-gene-evidence regime.
2. **The adaptive landscape must be supplied.** SCOUT *tests* a hypothesised regime partition
   (leaf labels + `ace` painting). It cannot *discover* where on the tree an adaptive event occurred.
   If you do not already know the niches, OUx is not available to you.
3. **Regimes are tied to observed cell state / tissue labels**, so "adaptive" is definitionally
   "differs between the groups I already drew".

Our model inverts all three:

- **event locations `z_b` are latent and shared across genes** — hundreds of genes each contributing
  weak evidence for the *same* branch, which is precisely the statistical strength SCOUT gives up;
- **`gamma_bg` gives sparse, gene-specific participation** in each event, so an event is defined by
  the genes that respond to it rather than by a pre-drawn label;
- the output is a *branch-level posterior* `P(z_b = 1 | Y)` plus the responding gene set — a
  discovery read-out SCOUT structurally cannot produce.

**The claim to test:** pooling weak evidence across genes recovers event branches in regimes where
per-gene model selection (SCOUT) is at or near chance.

## 4. What we must be careful about (honesty notes)

- **Do not strawman SCOUT.** Its OUx test, *given the correct regimes*, is a strong and appropriate
  test. Our comparison must either (a) give SCOUT the true regime labels and still beat it on
  localisation, or (b) be explicit that we are solving a task (de-novo event discovery) it was never
  designed for — and then the honest framing is "new capability", not "better accuracy".
- **The fair head-to-head** is therefore: SCOUT with the true leaf regimes vs. us with nothing but
  the tree and expression, scored on *gene-level* recovery (a task both can do). Anything else needs
  an explicit caveat.
- **Their smoothing is a real advantage** on noisy counts and is cheap to implement; we should either
  adopt an equivalent or model the noise explicitly (observation model), not ignore it.
- **The alpha/delta confound** (effect ~ (1 - e^{-alpha t}) * delta) is ours to worry about too, and is
  why gene-specific alpha_g needs hierarchical regularisation (roadmap Model 3).
