suppressMessages(library(ape))
source('/tmp/rvalid/fn.R')
set.seed(1)
tr <- read.tree(text="(((a:0.3,b:0.7):0.4,(c:0.2,d:0.9):0.6):0.5,((e:0.8,f:0.1):0.3,(g:0.45,h:0.25):0.55):0.35);")
n <- length(tr$tip.label)
X <- matrix(round(rnorm(n*3, 5, 2), 4), nrow=n, dimnames=list(tr$tip.label, c("g1","g2","g3")))
write.csv(X, '/tmp/rvalid/X.csv')
write.tree(tr, '/tmp/rvalid/tree.nwk')
for (k in c(2,3,5)) {
  S <- lineage_smooth(tr, X, k)
  write.csv(S, sprintf('/tmp/rvalid/S_k%d.csv', k))
}
cat("R done; tips:", paste(tr$tip.label, collapse=","), "\n")
