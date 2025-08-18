def setup_anndata(adata, tree):
    adata.uns['tree'] = tree
    return adata

def cut_tree(adata, min_cells=10):
    pass
