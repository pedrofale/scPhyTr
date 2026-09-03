/* SCOUT's per-gene question, answered with a posterior instead of a min-AICc argmax.
 *
 * For one gene, given a tree and a SUPPLIED regime painting, SCOUT fits three hypotheses and picks
 * the smallest AICc:
 *     BM1  neutral drift               mean = root state,        cov = sigma^2 C
 *     OU1  one global optimum          mean = theta,             cov = OU
 *     OUx  one optimum per regime      mean = W(alpha) theta,    cov = OU
 *
 * Stan cannot sample the discrete model indicator, so it is MARGINALISED:
 *     target += log_sum_exp(log_model_prior + loglik) ,
 * and softmax of the same vector, averaged over draws, is the Rao-Blackwellised P(model | y).
 * One fit per gene gives both the model probabilities and full posteriors for alpha/sigma/theta.
 *
 * Parameterisation choices that matter:
 *  - the OU scale is the STATIONARY SD s, not sigma; s is comparable across alpha and takes a
 *    prior on the same footing as the (standardised) data. sigma = s * sqrt(2 alpha).
 *  - alpha is parameterised by PHYLOGENETIC HALF-LIFE relative to tree height, hl = ln2/(alpha T).
 *    hl ~ 1 means "optimum reached over the tree's lifetime"; this avoids both degenerate ends.
 *  - each model carries its OWN parameters (product-space form), so its marginal likelihood is not
 *    contaminated by the others. Unused models' parameters simply revert to their priors.
 *
 * The OUx prior width s_theta IS the Occam penalty that replaces AICc's parameter count, so it is
 * exposed as data and must be swept, not fixed and forgotten.
 *
 * BM1 costs O(n) per gradient evaluation, not O(n^3): its covariance SHAPE C is data, so C is
 * eigendecomposed once in transformed data and y is rotated into that basis there too. Only the
 * scalars sigma_bm and tau then vary, so the log-determinant and quadratic form are elementwise.
 * The two OU models keep their own alpha (as OUwie does), so they each need a Cholesky per step.
 *
 * Tree structure never enters Stan: the regime weights come from a precomputed segment list
 * (tip, regime, start time, end time) along each root-to-tip path.
 */
data {
  int<lower=1> n;                       // tips
  int<lower=1> K;                       // supplied regimes
  vector[n] y;                          // one gene, standardised
  matrix[n, n] D;                       // patristic distance between tips
  matrix[n, n] C;                       // BM covariance shape (shared root-to-MRCA path length)
  vector[n] tip_depth;                  // root-to-tip time
  int<lower=0> n_seg;
  array[n_seg] int<lower=1, upper=n> seg_tip;
  array[n_seg] int<lower=1, upper=K> seg_reg;
  vector[n_seg] seg_s;                  // segment start time (from root)
  vector[n_seg] seg_e;                  // segment end time
  int<lower=1, upper=K> root_reg;
  real<lower=0> tree_height;

  int<lower=0, upper=1> use_tau;        // measurement-error term (replaces lineage smoothing)
  real<lower=0> s_theta;                // optimum prior sd  -- THE Occam knob
  real<lower=0> s_scale;                // prior sd for the BM/OU scales
  real<lower=0> s_tau;
  real<lower=0> sd_log_hl;              // lognormal sd for half-life / tree height
  vector[3] log_model_prior;            // BM1, OU1, OUx  (uniform 1/3 by default)
  int<lower=0, upper=3> fit_model;      // 0 = marginalise over models; 1/2/3 = that model alone
}

transformed data {
  matrix[n, n] I = diag_matrix(rep_vector(1.0, n));
  // BM1 in closed form: C = Q diag(lam) Q', and y is rotated once, here.
  vector[n] lam_C = eigenvalues_sym(C);
  matrix[n, n] Q_C = eigenvectors_sym(C);
  vector[n] Qty = Q_C' * y;
  vector[n] Qt1 = Q_C' * rep_vector(1.0, n);
  real n_log_2pi = n * log(2 * pi());
  // as alpha -> 0 the OU covariance exp(-alpha D) tends to the all-ones matrix, which is rank 1;
  // a fixed jitter keeps every Cholesky well defined across the whole prior range.
  matrix[n, n] JIT = diag_matrix(rep_vector(1e-8, n));
}

parameters {
  real theta_bm;
  real<lower=0> sigma_bm;

  real theta_ou1;
  real<lower=0> s_ou1;                  // stationary sd
  real<lower=1e-4, upper=1e4> hl_ou1;   // half-life / tree height (bounded to keep exp(-aD) finite)

  vector[K] theta_oux;
  real<lower=0> s_oux;
  real<lower=1e-4, upper=1e4> hl_oux;

  array[use_tau ? 3 : 0] real<lower=0> tau;
}

transformed parameters {
  real alpha_ou1 = log(2) / (hl_ou1 * tree_height);
  real alpha_oux = log(2) / (hl_oux * tree_height);
  vector[3] lp;
  {
    real t1 = use_tau ? square(tau[1]) : 0.0;
    real t2 = use_tau ? square(tau[2]) : 0.0;
    real t3 = use_tau ? square(tau[3]) : 0.0;

    // --- BM1, O(n) in the eigenbasis of C ---
    {
      vector[n] d = square(sigma_bm) * lam_C + (t1 + 1e-8);
      vector[n] qr = Qty - theta_bm * Qt1;
      lp[1] = -0.5 * (n_log_2pi + sum(log(d)) + dot_product(square(qr), inv(d)));
    }

    // --- OU1: root at its own optimum, so the tip mean is theta everywhere ---
    if (fit_model == 0 || fit_model == 2) {
      matrix[n, n] V = square(s_ou1) * exp(-alpha_ou1 * D) + t2 * I + JIT;
      lp[2] = multi_normal_cholesky_lpdf(y | rep_vector(theta_ou1, n), cholesky_decompose(V));
    } else lp[2] = 0;

    // --- OUx: tip mean is linear in the optima, W(alpha) from the segment list ---
    if (fit_model == 0 || fit_model == 3) {
      matrix[n, K] W = rep_matrix(0.0, n, K);
      matrix[n, n] V = square(s_oux) * exp(-alpha_oux * D) + t3 * I + JIT;
      for (j in 1:n_seg) {
        int i = seg_tip[j];
        W[i, seg_reg[j]] += exp(-alpha_oux * (tip_depth[i] - seg_e[j]))
                          - exp(-alpha_oux * (tip_depth[i] - seg_s[j]));
      }
      for (i in 1:n) W[i, root_reg] += exp(-alpha_oux * tip_depth[i]);
      lp[3] = multi_normal_cholesky_lpdf(y | W * theta_oux, cholesky_decompose(V));
    } else lp[3] = 0;
  }
}

model {
  theta_bm ~ normal(0, s_theta);
  theta_ou1 ~ normal(0, s_theta);
  theta_oux ~ normal(0, s_theta);
  sigma_bm ~ normal(0, s_scale);
  s_ou1 ~ normal(0, s_scale);
  s_oux ~ normal(0, s_scale);
  hl_ou1 ~ lognormal(0, sd_log_hl);
  hl_oux ~ lognormal(0, sd_log_hl);
  if (use_tau) tau ~ normal(0, s_tau);

  if (fit_model == 0)
    target += log_sum_exp(log_model_prior + lp);
  else
    target += lp[fit_model];
}

generated quantities {
  // Rao-Blackwellised posterior model probabilities -- meaningful only when fit_model == 0,
  // since a single-model fit skips the other two likelihoods and leaves their lp entries at 0.
  simplex[3] p_model = softmax(log_model_prior + lp);
  real sigma_ou1 = s_ou1 * sqrt(2 * alpha_ou1);
  real sigma_oux = s_oux * sqrt(2 * alpha_oux);
  real theta_spread = max(theta_oux) - min(theta_oux);
}
