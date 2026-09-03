#!/usr/bin/env bash
# Build a dedicated R environment that runs the ORIGINAL SCOUT (not our port).
#
# Design note: we do NOT `install.packages(SCOUT)` because SCOUT's NAMESPACE does `import(TedSim)`,
# so loading it as a package hard-requires TedSim -- which needs dyno/Bioconductor and is only used
# by their *simulation* helpers, not by model fitting. Instead we source their R files directly
# (skipping R/simulate.R). That still executes THEIR code, which is the point.
set -euo pipefail
ENV=scout-r
CONDA=~/miniconda3
R="$CONDA/envs/$ENV/bin/Rscript"

if [ ! -x "$R" ]; then
  "$CONDA/bin/conda" create -y -n $ENV -c conda-forge \
    r-base=4.4 r-ape r-corpcor r-nloptr r-future.apply r-remotes r-rcolorbrewer \
    r-data.table r-dplyr r-matrix r-phytools r-igraph
fi

"$R" -e '
repos <- "https://cloud.r-project.org"
need <- c("OUwie","paleotree","phylolm","castor","reshape2","progressr","stringr",
          "foreach","doParallel")
miss <- need[!sapply(need, requireNamespace, quietly=TRUE)]
if (length(miss)) { cat("installing:", paste(miss, collapse=", "), "\n")
                    install.packages(miss, repos=repos, Ncpus=4) }
ok <- sapply(c(need,"ape","corpcor","nloptr","future.apply","dplyr","Matrix"),
             requireNamespace, quietly=TRUE)
print(ok)
if (any(!ok)) { cat("MISSING:", paste(names(ok)[!ok], collapse=", "), "\n"); quit(status=1) }
cat("ALL SCOUT FITTING DEPENDENCIES PRESENT\n")'
