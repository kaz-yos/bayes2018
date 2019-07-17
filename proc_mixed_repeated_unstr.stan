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
  // Hyperparameteres for Sigma
  // Not done yet
  // Weight
  matrix[J,I] weight;
  // Whether to evaluate likelihood
  int<lower=0,upper=1> use_lik;
}

transformed data {

}

parameters {
  vector[J] Mu;
  cov_matrix[J] Sigma;
}

transformed parameters {

}

model {
  // Priors
  // Mean part
  for (j in 1:J) {
    target += normal_lpdf(Mu[j] | Mu_means[j], Mu_sds[j]);
  }
  // Covariance part
  // Inv-Wishart is typical, but not recommended?
  // Blank for now.

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
