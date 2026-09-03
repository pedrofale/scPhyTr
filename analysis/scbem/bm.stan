/* Single-gene Brownian motion on a lineage tree.
 *
 * Companion to analysis/scbem/bm_single_gene.tex and to the spec at
 * scPhyTr-markdown/methods/method-bm-single-gene-stan.md.
 *
 * The model is
 *
 *     y ~ Normal( m0 * 1 ,  sigma^2 C + diag(tau^2 w) + v0 * 11' )
 *
 * where C_ij is the root-to-MRCA path length shared by tips i and j, tau^2 is the
 * independent tip variance, w are per-tip weights (all 1 for one cell per tip;
 * 1/n_i when a tip is a collapsed group of n_i indistinguishable cells), and the
 * v0 * 11' term is what marginalising the root state contributes.
 *
 * Two engines compute that density:
 *   bm_prune_lpdf  -- Felsenstein pruning, O(n), used by the model block
 *   bm_dense_lpdf  -- the literal multivariate normal, O(n^3), the reference
 * Both are evaluated in generated quantities at every draw, so lp_diff is a
 * like-for-like correctness check at identical parameter values, for free.
 *
 * THE TREE NEVER ENTERS AS A TREE. Stan has no recursion and no ragged arrays, so
 * the caller passes flat arrays with the nodes ALREADY NUMBERED IN POST-ORDER:
 * tips are 1..n, internal nodes are n+1..2n-1, and every node's children have
 * smaller indices than the node itself. The recursion is then a forward loop.
 * The tree must be strictly binary; collapse unary chains and binarise polytomies
 * with zero-length edges before calling.
 *
 * stanc --warn-pedantic reports exactly three warnings, all saying that V_tot, h or
 * sigma2_c "has no priors".
 * All three are false positives: the priors sit inside an if (param_mode == ...) branch,
 * which the pedantic checker cannot see through. It says so itself. Do not chase it.
 */
functions {
  /* Felsenstein pruning. Messages carry a mean m and a variance V expressed in
   * units of sigma^2, so sigma^2 factors out of the quadratic form entirely.
   * Emits one contrast per internal node plus one at the root: n terms for n tips.
   */
  real bm_prune_lpdf(vector y, array[] int c1, array[] int c2, vector blen,
                     vector tau2_tip, real sigma2, real m0, real v0) {
    int n = num_elements(y);
    int N = 2 * n - 1;
    vector[N] m;
    vector[N] V;
    real q = 0;
    real logdet = 0;

    m[1:n] = y;
    V[1:n] = tau2_tip / sigma2;

    for (k in 1:(n - 1)) {
      int a = c1[k];
      int b = c2[k];
      real va = V[a] + blen[a];
      real vb = V[b] + blen[b];
      real w = va + vb;
      q += square(m[a] - m[b]) / w;
      logdet += log(w);
      // product form, not the harmonic one: stays finite when either child has
      // zero variance, which happens when tau = 0 meets a zero-length edge
      // introduced by binarising a polytomy.
      V[n + k] = va * vb / w;
      m[n + k] = (m[a] * vb + m[b] * va) / w;
    }
    {
      real vr = V[N] + v0 / sigma2;      // root prior, marginalised analytically
      q += square(m[N] - m0) / vr;
      logdet += log(vr);
    }
    return -0.5 * (n * log(2 * pi() * sigma2) + logdet + q / sigma2);
  }

  /* The same density, written out. Reference only. */
  real bm_dense_lpdf(vector y, matrix C, vector tau2_tip,
                     real sigma2, real m0, real v0) {
    int n = num_elements(y);
    matrix[n, n] S = sigma2 * C
                   + diag_matrix(tau2_tip)
                   + v0 * rep_matrix(1.0, n, n);
    return multi_normal_lpdf(y | rep_vector(m0, n), S);
  }
}

data {
  int<lower=2> n;                            // tips
  int<lower=3> N;                            // nodes, must equal 2n-1
  vector[n] y;                               // one gene at the tips
  array[n - 1] int<lower=1, upper=2 * n - 2> c1;   // children of internal node n+k
  array[n - 1] int<lower=1, upper=2 * n - 2> c2;
  vector<lower=0>[N] blen;                   // branch above each node; blen[N] unused
  vector<lower=0>[n] tip_w;                  // per-tip noise weights, 1 by default
  matrix[n, n] C;                            // dense covariance shape (reference)
  real<lower=0> Tbar;                        // mean root-to-tip length

  real m0;                                   // root prior mean
  real<lower=0> v0;                          // root prior variance (param_mode 0)

  int<lower=0, upper=1> use_tau;             // 0 turns the tip-noise term off
  int<lower=0, upper=1> engine;              // 0: pruning (default). 1: dense, for timing
  int<lower=0, upper=1> param_mode;          // 0: (V, h).  1: conjugate test config
  real<lower=0> k_root;                      // param_mode 1: v0 = sigma2 * k_root

  real mu_log_V;                             // prior: log V ~ normal(mu, sd)
  real<lower=0> sd_log_V;
  real<lower=0> h_a;                         // prior: h ~ beta(h_a, h_b)
  real<lower=0> h_b;
  real<lower=0> ig_a;                        // param_mode 1: sigma2 ~ inv_gamma(a, b)
  real<lower=0> ig_b;
}

transformed data {
  int n_V = param_mode == 0 ? 1 : 0;
  int n_C = param_mode == 1 ? 1 : 0;
}

parameters {
  // Conditionally sized so that no parameter is ever left to sample from its own
  // prior in a configuration that does not use it.
  array[n_V] real<lower=0> V_tot;            // total tip variance
  array[n_V] real<lower=0, upper=1> h;       // phylogenetic heritability
  array[n_C] real<lower=0> sigma2_c;         // conjugate-test rate
}

transformed parameters {
  real sigma2;
  real tau2;
  real v0_eff;
  if (param_mode == 0) {
    sigma2 = V_tot[1] * h[1] / Tbar;
    tau2 = use_tau ? V_tot[1] * (1 - h[1]) : 0.0;
    v0_eff = v0;
  } else {
    sigma2 = sigma2_c[1];
    tau2 = 0.0;                              // the conjugate configuration has no tip noise
    v0_eff = sigma2_c[1] * k_root;           // root prior scaled with the rate
  }
}

model {
  if (param_mode == 0) {
    V_tot[1] ~ lognormal(mu_log_V, sd_log_V);
    h[1] ~ beta(h_a, h_b);
  } else {
    sigma2_c[1] ~ inv_gamma(ig_a, ig_b);
  }
  // Identical densities; the flag exists so the two can be timed against each other.
  // Written with target += rather than two tilde statements, which stanc would
  // (falsely) flag as putting y on the left of more than one tilde.
  target += engine == 0
    ? bm_prune_lpdf(y | c1, c2, blen, tau2 * tip_w, sigma2, m0, v0_eff)
    : bm_dense_lpdf(y | C, tau2 * tip_w, sigma2, m0, v0_eff);
}

generated quantities {
  // Evaluated at the SAME draw, so the difference isolates the engines rather
  // than two separate fits. This is the correctness check.
  real lp_prune = bm_prune_lpdf(y | c1, c2, blen, tau2 * tip_w, sigma2, m0, v0_eff);
  real lp_dense = bm_dense_lpdf(y | C, tau2 * tip_w, sigma2, m0, v0_eff);
  real lp_diff = lp_prune - lp_dense;
}
