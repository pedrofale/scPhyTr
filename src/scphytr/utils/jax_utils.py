import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular


def build_cholesky(L_params, min_diag=1e-6):
    # lower triangle free, diagonal forced positive
    diag = jax.nn.softplus(jnp.diag(L_params)) + min_diag
    L = jnp.tril(L_params, -1) + jnp.diag(diag)
    return L  # L is lower-tri with positive diag

def mvn_kron_logpdf_traitmajor_with_L(y, means, Lk, Lc, n_species):
    """
    y: shape (n*p,) flattened in TRAIT-major order:
       [t1(s1..sn), t2(s1..sn), ...]
    means: (p,)
    Lk: (p,p) lower-tri Cholesky of trait covariance K (K = Lk @ Lk.T)
    Lc: (n,n) species Cholesky
    n_species: n
    """
    p = means.shape[0]
    n = n_species

    # Mean vector in trait-major order: [m1]*n + [m2]*n + ...
    a = jnp.repeat(means, n)

    # Reshape centered data to (traits × species)
    Z = (y - a).reshape(p, n)           # p × n

    # Quadratic term via triangular solves:
    # U = Lk^{-1} Z  (traits whitening)
    U = solve_triangular(Lk, Z, lower=True)          # p × n
    # V = Ls^{-1} U^T (species whitening)
    V = solve_triangular(Lc, U.T, lower=True)        # n × p
    quad = jnp.sum(V * V)

    # Log-determinant using Cholesky diagonals
    logdetK = 2.0 * jnp.sum(jnp.log(jnp.diag(Lk)))
    logdetS = 2.0 * jnp.sum(jnp.log(jnp.diag(Lc)))
    d = n * p

    return -0.5 * (quad + n * logdetK + p * logdetS + d * jnp.log(2.0 * jnp.pi))

@jax.jit
def sample_mvnormal_kron_traitmajor(key, Ls, Lk, mean):
    """
    Sample x ~ N(mean, K ⊗ C) without forming the Kronecker.
    TRAIT-major layout: [trait1(all species), trait2(all species), ...].

    Args:
      key:  PRNGKey
      Ls:   (n,n) lower-tri Cholesky of C  (species covariance)
      Lk:   (p,p) lower-tri Cholesky of K  (trait   covariance)
      mean: (p,) trait means (shared across species) OR full (p*n,) vector (trait-major)

    Returns:
      x: (p*n,) sample in trait-major order
    """
    n = Ls.shape[0]
    p = Lk.shape[0]

    # Standard normal noise shaped as (traits × species)
    Z = jax.random.normal(key, (p, n))

    # Apply trait factor, then species factor (no kron)
    Z = Lk @ Z             # (p, n)
    Z = Z @ Ls.T            # (p, n)

    x = Z.reshape(p * n)   # trait-major flatten

    # Add mean (expand if given as per-trait)
    # if mean.ndim == 1 and mean.shape[0] == p:
    mean_vec = jnp.repeat(mean, n)     # [m1]*n, [m2]*n, ...
    # else:
        # mean_vec = mean                    # assume full (p*n,) in trait-major order
    return x + mean_vec