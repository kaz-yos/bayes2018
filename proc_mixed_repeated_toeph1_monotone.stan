functions {

}

data {
  // Number of clusters
  int<lower=1> I;
  // Number of balanced observations within each cluster
  int<lower=1> J;
  // https://mc-stan.org/docs/2_18/stan-users-guide/multivariate-hierarchical-priors-section.html
  // Hyperparameteres for Mu
  real Mu_means[J];
  real<lower=0> Mu_sds[J];
  // Hyperparameters for sigma
  real sigma_cauchy_scale[J];
  // Hyperparameter for rho's
  real<lower=0> logit_rho_raw_means[J-1];
  real<lower=0> logit_rho_raw_sds[J-1];
  // Weight
  matrix[J,I] weight;
  // Whether to evaluate likelihood
  int<lower=0,upper=1> use_lik;
  // Print debug flag
  int<lower=0,upper=1> debug;
}



transformed data {

}

parameters {
  vector[J] Mu;
  // For SD
  real<lower=0> sigma[J];
  // For Corr. Increasing. Need to reverse.
  ordered[J-1] logit_rho_raw;
}

transformed parameters {
  real<lower=-1,upper=1> rho[J-1];
  matrix[J,J] Corr;
  matrix[J,J] Sigma;
  // Populate the correlation matrix with the Toeplitz structure.
  for (j in 1:(J-1)) {
    rho[j] = 2 * inv_logit(logit_rho_raw[J - j]) - 1;
  }
  if (debug) {
    print(rho);
  }
  for (i in 1:J) {
    for (j in 1:J) {
      if (i == j) {
        // Correlation with self
        Corr[i,j] = 1;
      } else {
        // Otherwise, choose the right rho.
        Corr[i,j] = rho[abs(i-j)];
      }
    }
  }

  // Cov = daig(SDs) * Corr * daig(SDs)
  // https://mc-stan.org/docs/2_19/functions-reference/diagonal-matrix-functions.html
  // https://mc-stan.org/docs/2_19/reference-manual/covariance-matrices-1.html
  Sigma = diag_matrix(to_vector(sigma)) * Corr * diag_matrix(to_vector(sigma));
}

model {
  // Priors
  // Mean part
  for (j in 1:J) {
    // Mean part
    target += normal_lpdf(Mu[j] | Mu_means[j], Mu_sds[j]);
    // Variance (SD) part
    // https://mc-stan.org/docs/2_19/functions-reference/cauchy-distribution.html
    target += cauchy_lpdf(sigma[j] | 0, sigma_cauchy_scale[j]);
  }
  // Corr part
  for (j in 1:(J-1)) {
    target += normal_lpdf(logit_rho_raw[j] | logit_rho_raw_means[j], logit_rho_raw_sds[j]);
  }

  // Likelihood
  if (use_lik == 1) {
    // Loop over clusters
    for (i in 1:I) {
      // Within each cluster
      target += multi_normal_lpdf(weight[,i] | Mu, Sigma);
    }
  }
}

generated quantities {
  // For loo
  vector[I] log_lik;
  // For posterior predictive checks
  matrix[J,I] weight_rep;

  for (i in 1:I) {
    // Observation level log likelihood
    log_lik[i] = multi_normal_lpdf(weight[,i] | Mu, Sigma);
    // Prediction
    weight_rep[,i] = multi_normal_rng(Mu, Sigma);
  }
}
