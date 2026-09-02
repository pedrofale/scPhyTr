#!/usr/bin/env Rscript
# Run the ORIGINAL SCOUT (Stuart & McKenna 2025) on a counts CSV + Newick tree.
#
# We source SCOUT's own R sources rather than installing the package, because its NAMESPACE does
# `import(TedSim)` and TedSim is needed only by their *simulation* helpers, not by model fitting.
# Everything executed here is their code.
#
# Usage:
#   Rscript run_scout.R --counts F.csv --tree T.nwk --out DIR [--regimes BM1,OU1,OUM]
#                       [--method SM|EM|MTF] [--species-key species] [--smoothing-k 8]
#                       [--cores 4] [--scale-tree] [--no-normalize]
LIB <- Sys.getenv("SPARSEOU_RLIB", "/Users/pedroferreira/projects/cce/sparseou/external/Rlib")
if (dir.exists(LIB)) .libPaths(c(LIB, .libPaths()))

suppressMessages({
  library(ape); library(dplyr); library(stringr); library(corpcor); library(nloptr)
  library(future.apply); library(paleotree); library(phylolm); library(OUwie); library(progressr)
})

args <- commandArgs(trailingOnly = TRUE)
getarg <- function(flag, default = NULL) {
  i <- which(args == flag); if (length(i) == 0) return(default); args[i + 1]
}
hasflag <- function(flag) any(args == flag)

counts_file <- getarg("--counts");  tree_file <- getarg("--tree");  outdir <- getarg("--out")
if (is.null(counts_file) || is.null(tree_file) || is.null(outdir))
  stop("--counts, --tree and --out are required")
regimes  <- strsplit(getarg("--regimes", "BM1,OU1,OUM"), ",")[[1]]
method   <- getarg("--method", "SM")
species  <- getarg("--species-key", "species")
smooth_k <- getarg("--smoothing-k", NULL); if (!is.null(smooth_k)) smooth_k <- as.numeric(smooth_k)
cores    <- as.integer(getarg("--cores", "1"))
testid   <- getarg("--testid", "run")
scaleT   <- hasflag("--scale-tree")
normalize <- !hasflag("--no-normalize")

SRC <- getarg("--scout-src", file.path(dirname(normalizePath(sub("--file=", "",
        grep("--file=", commandArgs(FALSE), value = TRUE)[1]))), "..", "external", "SCOUT", "R"))
for (f in c("SCOUT_EM_utils.R", "SCOUT_EM.R", "analysis_utils.R", "main.R"))
  source(file.path(SRC, f))                     # NB: simulate.R skipped (needs TedSim)

dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
logfile <- file.path(outdir, "scout.log")
cat(sprintf("SCOUT: counts=%s tree=%s method=%s regimes=%s k=%s\n", counts_file, tree_file,
            method, paste(regimes, collapse = "/"), ifelse(is.null(smooth_k), "NULL", smooth_k)))

counts <- read.csv(counts_file, row.names = 1)
tree   <- ape::read.tree(tree_file)

# SCOUT()'s smoothing_k branch references undefined `ss[i,'smoothing_k']` (a leftover from a
# samplesheet-driven script), so passing a numeric smoothing_k to SCOUT() errors. Their default
# path (method='SM' with smoothing_k=NULL -> k=8) is fine. For any other k we call their
# formatSCOUT/runSCOUT directly, which is what SCOUT() itself does.
use_wrapper <- is.null(smooth_k) || (method == "SM" && smooth_k == 8)

if (use_wrapper) {
  res <- SCOUT(counts.file = counts_file, tree.file = tree_file, results_dir = outdir,
               regimes = regimes, species_key = species, method = method,
               testid = testid, normalize = normalize, scale_tree = scaleT,
               cores = cores, logfile = logfile, verbose = TRUE)
} else {
  skipE <- method != "EM"; skipT <- method == "SM"
  idata <- formatSCOUT(tree_path = tree, metadata_path = counts, species_key = species,
                       anc_infer = "ape", outpath = outdir, regimes = regimes,
                       normalize = normalize, smoothing_k = smooth_k, logfile = logfile)
  full <- runSCOUT(idata, fixed.root = FALSE, M_only = skipE, runid = testid, skipTau = skipT,
                   scaleHeight = scaleT, cores = cores, logfile = logfile)
  hist <- extract_history_grid_search(full)
  if (!"converge" %in% colnames(hist)) hist$converge <- NA
  hist$dataset <- testid
  ann <- annotate_history(hist, datasetid = "dataset")
  write.csv(ann, sprintf("%s/%s_all_genes_full_history.csv", outdir, testid))
  best <- ann %>% filter(delta_AIC == 0)
  write.csv(best, sprintf("%s/%s_annotated_best_fit.csv", outdir, testid))
  res <- list(SCOUT_class = best)
}
cat("SCOUT finished. Outputs in", outdir, "\n")
print(utils::head(as.data.frame(res$SCOUT_class), 5))
