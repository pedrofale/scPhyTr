#!/usr/bin/env Rscript
# Bridge to the REAL EvoGeneX R package (not the Python surrogate). Reads a long-format
# expression table (gene, species, replicate, exprval), a tree, and a two-regime file,
# fits replicate-aware Brownian motion and the two-regime EvoGeneX OU model per gene, and
# writes per-gene log-likelihoods + the adaptive (OU2-vs-BM) likelihood-ratio p-value.
#
# Usage: Rscript evogenex_bridge.R <long.csv> <tree.nwk> <regime.csv> <out.csv>
suppressMessages({library(dplyr); library(tidyr); library(EvoGeneX)})

args <- commandArgs(trailingOnly = TRUE)
long_csv <- args[1]; tree_nwk <- args[2]; regime_csv <- args[3]; out_csv <- args[4]

dat <- read.csv(long_csv, stringsAsFactors = FALSE)
genes <- unique(dat$gene)

brown <- Brown(); brown$setTree(tree_nwk)
evog <- EvoGeneX(); evog$setTree(tree_nwk); evog$setRegimes(regime_csv)

bm_dof <- 3      # sigma.sq, theta, gamma
ou2_dof <- 5     # alpha, sigma.sq, 2 thetas, gamma

res <- data.frame()
for (g in genes) {
  d <- dat[dat$gene == g, c("species", "replicate", "exprval")]
  out <- tryCatch({
    bm  <- brown$fit(d, gamma = 0.01, format = "tall")
    ou2 <- evog$fit(d, alpha = 0.1, gamma = 0.01, format = "tall")
    lr  <- 2 * (ou2$loglik - bm$loglik)
    p   <- 1 - pchisq(max(lr, 0), ou2_dof - bm_dof)
    data.frame(gene = g, bm_loglik = bm$loglik, ou2_loglik = ou2$loglik,
               alpha = ou2$alpha, lr = lr, p = p,
               # EvoGeneX adaptive call by AIC (matches scPhyTr's criterion)
               adaptive_aic = as.integer((2 * ou2_dof - 2 * ou2$loglik) <
                                         (2 * bm_dof - 2 * bm$loglik)))
  }, error = function(e) {
    data.frame(gene = g, bm_loglik = NA, ou2_loglik = NA, alpha = NA,
               lr = NA, p = NA, adaptive_aic = NA)
  })
  res <- rbind(res, out)
}
res$padj <- p.adjust(res$p, method = "fdr")
write.csv(res, out_csv, row.names = FALSE)
cat("EvoGeneX: fit", nrow(res), "genes;",
    sum(res$adaptive_aic, na.rm = TRUE), "adaptive (AIC),",
    sum(res$padj < 0.05, na.rm = TRUE), "adaptive (LRT FDR<0.05)\n")
