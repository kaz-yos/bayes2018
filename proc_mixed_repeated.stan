functions {

}

data {
  // Total number of observations
  int<lower=1> N;
  // Number of clusters
  int<lower=1> I;
  // Number of balanced observations within each cluster
  int<lower=1> J;
  // Whether to evaluate likelihood
  int<lower=0,upper=1> use_lik;
  // https://mc-stan.org/docs/2_18/stan-users-guide/multivariate-hierarchical-priors-section.html
  // Hyperparameteres for Mu
  real Mu_means[J];
  real<lower=0> Mu_sds;
  // Hyperparameteres for Sigma
  real
  // Cluster ID
  int<lower=1> id[N];
  // Weight
  real weight[J,I];
}

transformed data {

}

parameters {
  vector[J] Mu;
  cov_matrix[J,J] Sigma;
}

transformed parameters {

}

model {
  // Priors
  // Mean part
  for (j in 1:J) {
    target += normal_lpdf(Mu | Mu_means[j], Mu_sds[j]);
  }

  // Likelihood
  if (use_lik == 1) {
    // Loop over clusters
    for (i in 1:I) {
      target += normal_lpdf(weight[,i] | Mu, Sigma);
    }
  }
}

generated quantities {
  // For loo
  vector[N] log_lik;
  // For posterior predictive checks
  real y_rep[N];

  for (i in 1:N) {
    // Observation level log likelihood
    log_lik[i] = ;
    // Prediction
    y_rep[i] = ;
  }
}
