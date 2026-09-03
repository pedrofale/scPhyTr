# Reading a tree: regime paintings and expression overlays

Two questions get asked of a lineage tree once there is a fit on it. *Which regime does this branch
sit in?* — a categorical label spread over the topology. And *what is this gene doing across the
tips?* — one or more continuous tracks read against that topology. `scphytr.pl` answers the first
with [`pl.plot_tree`](#painting-a-tree-by-regime) and the second with
[`pl.expression_tree`](#expression-overlays).

## Painting a tree by regime

A **painting** assigns a regime to every branch, given regimes only at the tips. `pl.plot_tree`
takes the tip labels and colours each branch with the regime of the tips below it:

```python
import scphytr as ph

ph.pl.plot_tree(tree, regimes=leaf_regime, tip_strip=True, label_leaves=False)
```

`regimes` is either a `{tip name -> label}` mapping or a sequence in the tree's own tip order. The
mapping is safer — it does not depend on which tip order you happen to be holding.

**A branch above a regime split belongs to no regime, and is drawn grey.** This is the one design
decision in the function worth arguing about, and it is deliberate. Attributing the branch above a
split to one of the two sides is exactly the confusion that a clade-coherence check exists to
expose: when a regime *is* a clade, its design column is a scaled clade indicator, and a model
cannot separate adaptation to the regime from inheritance from the ancestor. A picture that paints
that branch anyway hides the very thing you are looking for. `pl.paint_from_leaves` returns the
painting on its own, with `None` at those nodes, if you want to count them rather than look at them.

### Parameters

| Parameter | Default | What it controls |
| --- | --- | --- |
| `tree` | — | A [`Tree`](01_methods.md), a bare ete3 node, the array tree from `scphytr.modes`, or an `AnnData` carrying `uns['tree']`. |
| `regimes` | `None` | Tip labels, as a mapping or a sequence in tip order. Mutually exclusive with `node_values` / `color`. |
| `palette` | `None` | `{label -> colour}`. Built with `pl.regime_palette` when absent; pass one to hold colours steady across panels. |
| `regime_cmap` | `"tab10"` | Colormap the palette is drawn from when `palette` is absent. Separate from `cmap`. |
| `tip_strip` | `False` | Draw a column of tip colours past the tips. |
| `legend` | `True` | Legend of regimes with their tip counts. |
| `node_values` | `None` | `{node -> float}` for the continuous read-out. Branches without a value are grey. |
| `color` | `None` | With an `AnnData`, a gene in `var_names` or an `obs` column to colour tips by. |
| `cmap` | `"viridis"` | Colormap for the **continuous** read-out. Unused when `regimes` is set. |
| `vmin`, `vmax` | `None` | Continuous colour limits; taken from the data when absent. |
| `label_leaves` | `True` | Write each tip's name. Turn this off past ~100 tips. |
| `linewidth` | `2.2` | Branch width. Drop to well under 1 on a tree of several hundred tips. |
| `ax`, `title`, `cbar_label` | `None`, `None`, `"value"` | Axes to draw into, title, colourbar label. |

Returns the matplotlib `Axes`.

`tip_strip=True` matters more than it sounds. Past a few hundred tips the branches are thinner than
the gaps between them and the blocks of colour stop being legible; the strip is a solid column and
stays readable at any tip count.

To keep one label the same colour across several panels, build the palette once over the union:

```python
pal = ph.pl.regime_palette(list(coarse_labels) + list(fine_labels))
ph.pl.plot_tree(tree, regimes=coarse, palette=pal, ax=axes[0])
ph.pl.plot_tree(tree, regimes=fine,   palette=pal, ax=axes[1])
```

`regime_palette(regimes, cmap="tab10")` assigns colours in first-seen order and de-duplicates, so a
label keeps its colour whatever else is in the list.

### The model-fitting counterpart

`pl.paint_from_leaves` is for drawing. The painting the *models* consume is
`scphytr.modes.baseline.paint_regimes(tree, leaf_regime, mixed="root")`, which returns integer codes
over the array tree. The two agree except at nodes whose tips span several regimes: `mixed="root"`
(the default, and what every fit wants) gives those the root regime so the painting covers every
branch, while `mixed=None` codes them `-1`, which is what the plot needs. Both are a deterministic
parsimony-style rule, **not** a maximum-likelihood ancestral reconstruction — on trees where regimes
are clade-coherent they coincide, and on real data where they are not, swap in a proper Mk
reconstruction. See [mode detection](06_mode_detection.md) for where these paintings are used.

## Expression overlays

One branch colour shows one variable. To read several at once — a few genes, against a painting,
against a covariate — put a colour strip per variable beside the tips:

```python
ph.pl.expression_tree(adata, ["Mki67", "Sox2", "cell_type"])
```

The strips are drawn by [`cassiopeia.pl.plot_matplotlib`](https://cassiopeia-lineage.readthedocs.io);
`scphytr` does not reimplement them. What it adds is resolving `keys` against an `AnnData` onto the
tips, and labelling the strips — cassiopeia leaves them unnamed, which makes a multi-gene panel
impossible to read.

Keys are genes in `var_names` (mean `log1p` expression over the cells at each tip) or `obs` columns
(numeric ones averaged, categorical ones taking the commonest label at the tip). Values that are not
in an `AnnData` at all go in through `cell_meta`, a frame indexed by tip name:

```python
meta = pd.DataFrame({"regime": leaf_regime}, index=tips).join(log_cpm[genes])
ph.pl.expression_tree(tree, ["regime"] + genes, cell_meta=meta)
```

Passing the painting in as one of the keys is the point of the function: a gene's expression can
then be read against the regime it sits in, on the same tips.

### Parameters

| Parameter | Default | What it controls |
| --- | --- | --- |
| `tree` | — | Anything `plot_tree` accepts, including an `AnnData` with `uns['tree']`. |
| `keys` | — | One key or a list. Genes, `obs` columns, or `cell_meta` columns. One strip each, in this order, nearest strip first. |
| `adata` | `None` | Where `keys` are looked up when `tree` is not itself an `AnnData`. |
| `cell_meta` | `None` | A frame indexed by tip name, used verbatim instead of `adata`. |
| `orient` | `"right"` | Tree direction, passed to cassiopeia (`"up"`, `"down"`, `"left"`, `"right"`, or an angle). |
| `continuous_cmap` | `"viridis"` | Numeric keys. Expression is the archetypal one, so this matches the project's expression convention. |
| `categorical_cmap` | `"tab10"` | Categorical keys. |
| `label_strips` | `True` | Name each strip. Only meaningful for `orient` `"left"`/`"right"`; ignored otherwise. |
| `figsize` | `(8.0, 9.0)` | Figure size. |
| `add_root` | `True` | Draw the root edge. |
| `title` | `None` | Axes title. |
| `**kwargs` | — | Forwarded to `cassiopeia.pl.plot_matplotlib` (`vmin`, `vmax`, `colorstrip_width`, …). |

Returns the matplotlib `Axes`.

Cassiopeia decides per column whether a key is continuous or categorical. A categorical key with
many levels gets no legend from either side — prefer a small number of levels, or recode.

## Which trees work

`pl.plot_tree`, `pl.expression_tree` and [`pl.rate_tree`](01_methods.md) all accept the same things:
a `scphytr.Tree`, a bare ete3 node, an `AnnData` carrying `uns['tree']`, or the array-backed tree
from `scphytr.modes`. The last is converted through `Tree.to_newick()`, which round-trips with
`Tree.from_newick` and keeps internal node names — those names are what lets a painting or a
per-branch value line back up with the tree. See `modes/_adapt.py` for why a second tree
representation exists at all.

## Honest limits

- **`expression_tree` needs an optional dependency.** `cassiopeia-lineage`, imported lazily, as in
  `scphytr.simulation`. PyPI's build is currently broken; install from source:
  `pip install "cassiopeia-lineage @ git+https://github.com/YosefLab/Cassiopeia.git"`. Nothing else
  on this page needs it.
- **Strip labels assume a horizontal tree.** They are recovered from the drawn patches, which only
  works for `orient="left"`/`"right"`; other orientations draw the strips unlabelled.
- **The painting is parsimony, not inference.** It reports what the tips imply, with no model and no
  uncertainty. It is a description of the labels you supplied, not an ancestral state estimate.
- **`plot_tree` is one variable at a time.** Branch colour carries either a painting or a continuous
  value, never both; `expression_tree` is the many-variable view, and it puts everything at the tips
  rather than along the branches.
- **Tip-level only for expression.** Both functions read expression at the tips. Neither draws a
  reconstructed value along internal branches.
