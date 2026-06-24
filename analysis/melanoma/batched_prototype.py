"""Proof-of-concept: batch the scalar tree-Laplace over the *gene* axis.

The per-gene scan is bottlenecked by Python loops over tree nodes, repeated for
every optimizer/Newton evaluation of every gene. But those loops are identical
across genes -- only the per-node scalars differ. Carrying a gene axis turns each
per-node scalar op into a vectorized ``(G,)`` op, so the Python loop is paid
*once* for all G genes. This prototype implements the batched BM Laplace marginal
(pure-Poisson summed counts) and shows it matches the per-gene path and that G
genes cost ~the same wall-time as one.

This is the engine that makes a transcriptome-wide *per-gene* model-selection
scan feasible; it is complementary to phylogenetic factor analysis (which shares
k factors across genes for the latent representation).
"""
import time
import numpy as np

from analysis.melanoma.load import load_counts, tree_leaves, load_tree
from scphytr.inference.laplace import MultiCellPoissonObservation, PoissonObservation
from scphytr.inference.tree_laplace import latent_tree_laplace_marginal


def build_topology(tree):
    """Static arrays from the tree: postorder parents, branch lengths, leaf rows."""
    post = list(tree.root.traverse("postorder"))
    index = {nd: i for i, nd in enumerate(post)}
    N = len(post)
    parent = np.full(N, -1, int)
    v = np.zeros(N)
    is_leaf = np.zeros(N, bool)
    for nd in post:
        i = index[nd]
        v[i] = nd.dist
        if nd.up is not None:
            parent[i] = index[nd.up]
        is_leaf[i] = nd.is_leaf()
    root_i = index[tree.root]
    leaf_rows = np.array([index[l] for l in tree.root.get_leaves()])
    # map tree.get_leaf_names() order -> postorder row
    name_row = {l.name: index[l] for l in tree.root.get_leaves()}
    leaf_order = np.array([name_row[n] for n in tree.phylotree.get_leaf_names()])
    return dict(N=N, parent=parent, v=v, is_leaf=is_leaf, root_i=root_i,
                leaf_order=leaf_order)


def batched_bm_marginal(topo, Y, S, sigma2, mu, n_newton=25):
    """BM Laplace marginal for G genes at once (pure-Poisson summed counts).

    Y, S : (n_leaves, G) summed counts and offsets. sigma2, mu : (G,).
    Returns (G,) marginal log-likelihoods. One Python loop over nodes per Newton
    step / per quantity; everything inside is vectorized over the gene axis.
    """
    N, parent, v = topo["N"], topo["parent"], topo["v"]
    root_i, leaf_order = topo["root_i"], topo["leaf_order"]
    G = Y.shape[1]
    invVu = np.where(v > 0, 1.0 / np.where(v > 0, v, 1.0), 0.0)     # 1/branch (unit)
    invV = invVu[:, None] / sigma2[None, :]                          # (N, G) = 1/(v sigma2)
    order = list(range(N))                                           # postorder indices

    # latent at every node; leaves seeded from data, internals from gene mean
    Z = np.repeat(mu[None, :], N, axis=0).astype(float)
    Z[leaf_order] = np.log((Y + 0.5) / (S + 1e-9))

    def eliminate(Wleaf, rhs):
        """Solve (Q + W) x = rhs and return (x, logdetH), batched over genes.

        Q is the BM tree precision (root tied to mu via invV at the root edge);
        W is the Poisson curvature at the leaves. One postorder + one preorder
        pass; per-node arrays are (G,).
        """
        d = np.zeros((N, G))
        # diagonal: own edge + children edges (+ leaf curvature)
        for i in order:
            d[i] += invV[i]
            pa = parent[i]
            if pa >= 0:
                d[pa] += invV[i]            # parent gets child's edge precision
        d[leaf_order] += Wleaf
        rr = rhs.copy()
        logdet = np.zeros(G)
        dd = d.copy()
        for i in order:                      # postorder elimination
            pa = parent[i]
            logdet += np.log(dd[i])
            if pa >= 0:
                off = -invV[i]               # coupling to parent
                dd[pa] -= off * off / dd[i]
                rr[pa] -= off * rr[i] / dd[i]
        x = np.zeros((N, G))
        for i in reversed(order):            # preorder back-substitution
            pa = parent[i]
            if pa < 0:
                x[i] = rr[i] / dd[i]
            else:
                x[i] = (rr[i] + invV[i] * x[pa]) / dd[i]
        return x, logdet

    # Newton on the negative log joint (Poisson leaves + Gaussian tree prior)
    for _ in range(n_newton):
        eta = Z[leaf_order]
        rate = S * np.exp(eta)
        gobs = np.zeros((N, G)); gobs[leaf_order] = Y - rate
        Wleaf = rate
        # prior gradient Q (Z - m): edge differences
        gpri = np.zeros((N, G))
        for i in order:
            pa = parent[i]
            if pa >= 0:
                dzi = (Z[i] - Z[pa]) * invV[i]
                gpri[i] += dzi
                gpri[pa] -= dzi
            else:
                gpri[i] += (Z[i] - mu) * invV[i]     # root edge to ancestral mu
        grad = gobs - gpri
        step, _ = eliminate(Wleaf, grad)
        Z = Z + step
        if np.max(np.abs(step)) < 1e-8:
            break

    # marginal = loglik + log N(Z;m,Q^-1) + 0.5 log|Q| - 0.5 log|Q+W|
    eta = Z[leaf_order]
    rate = S * np.exp(eta)
    loglik = np.sum(Y * eta - rate, axis=0)          # drop const gammaln (cancels in comparison)
    quad = np.zeros(G)
    logdetQ = np.zeros(G)
    for i in order:
        pa = parent[i]
        if pa >= 0:
            quad += invV[i] * (Z[i] - Z[pa]) ** 2
        else:
            quad += invV[i] * (Z[i] - mu) ** 2
        logdetQ += np.log(invV[i])
    Wleaf = rate
    _, logdetH = eliminate(Wleaf, np.zeros((N, G)))
    return loglik - 0.5 * quad + 0.5 * logdetQ - 0.5 * logdetH


def main():
    X, genes, clone, sf = load_counts()
    leaves = tree_leaves(); leaf_of = {n: k for k, n in enumerate(leaves)}
    idx = np.array([leaf_of[c] for c in clone]); nL = len(leaves)
    tree = load_tree()
    topo = build_topology(tree)

    # summed counts/offsets per leaf (pure Poisson), for a batch of genes
    order_g = np.argsort(X.sum(0))[::-1]
    pick = [g for g in order_g if (X[:, g] > 0).mean() > 0.6][:300]

    def summed(g):
        Y = np.zeros(nL); S = np.zeros(nL)
        np.add.at(Y, idx, X[:, g]); np.add.at(S, idx, sf)
        return Y, S
    Ys = np.stack([summed(g)[0] for g in pick], 1)      # (nL, G)
    Ss = np.stack([summed(g)[1] for g in pick], 1)
    # align rows to postorder leaf order
    Ys = Ys[np.argsort(topo["leaf_order"].argsort())]  # identity: leaf_order already maps
    sigma2 = np.full(len(pick), 0.05); mu = np.log((Ys.sum(0) + 1) / (Ss.sum(0) + 1))

    # --- correctness vs per-gene path (first 3 genes) ---
    bm = batched_bm_marginal(topo, Ys[:, :3], Ss[:, :3], sigma2[:3], mu[:3])
    print("correctness vs per-gene latent_tree_laplace_marginal:")
    for j in range(3):
        obs = PoissonObservation(Ys[:, j], Ss[:, j])
        ref = latent_tree_laplace_marginal(tree, obs, 0.0, mu[j], sigma2[j], root_value=mu[j])
        ref_noconst = ref + np.sum(np.log(np.arange(1, 1)))  # gammaln const dropped in batched
        # add back gammaln const to batched for fair compare
        from scipy.special import gammaln
        b = bm[j] - np.sum(gammaln(Ys[:, j] + 1.0))
        print(f"  gene {j}: batched={b:.3f}  per-gene={ref:.3f}  diff={abs(b-ref):.2e}")

    # --- speed: G genes at once vs the per-node loop cost ---
    for G in (1, 50, 300):
        t = time.time()
        for _ in range(3):
            batched_bm_marginal(topo, Ys[:, :G], Ss[:, :G], sigma2[:G], mu[:G])
        dt = (time.time() - t) / 3
        print(f"batched marginal, G={G:3d}: {dt*1000:6.1f} ms total  "
              f"({dt/G*1000:.3f} ms/gene)")


if __name__ == "__main__":
    main()
