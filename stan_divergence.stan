data {
  // Number of observations
  int<lower=1> N;
  // Whether to evaluate likelihood
  int<lower=0,upper=1> use_lik;
  //
  real<lower=0> sigma_sd;
  // Number of groups J
  int<lower=1> J;
}

transformed data {

}

parameters {
  real<lower=0> sigma;
  vector[J] mu;
}

transformed parameters {

}

model {
  // Priors
  target += normal_lpdf(sigma | 0, sigma_sd);
  for (j in 1:J) {
    target += normal_lpdf(mu[j] | 0, sigma);
  }
  // Likelihood
  if (use_lik == 1) {
    target += normal_lpdf(y[i,j] | mu[j], 1);
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
