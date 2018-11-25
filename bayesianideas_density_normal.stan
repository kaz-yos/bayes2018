data {
    // Hyperparameters
    real alpha;
    real beta;
    real m;
    real s_squared;

    // Define variables in data
    // Number of observations (an integer)
    int<lower=0> n;
    // Outcome (a real vector of length n)
    real y[n];
}

transformed data {
    real s;
    s <- sqrt(s_squared);
}

parameters {
    // Define parameters to estimate
    // Population mean (a real number)
    real mu;
    // Population variance (a positive real number)
    real<lower=0> sigma_squared;
}

transformed parameters {
    // Population standard deviation (a positive real number)
    real<lower=0> sigma;
    // Standard deviation (derived from variance)
    sigma <- sqrt(sigma_squared);
}

model {
    // Prior part of Bayesian inference
    // Mean
    mu ~ normal(m, s);
    // sigma^2 has inverse gamma (alpha = 1, beta = 1) prior
    sigma_squared ~ inv_gamma(alpha, beta);

    // Likelihood part of Bayesian inference
    // Outcome model N(mu, sigma^2) (use SD rather than Var)
    y ~ normal(mu, sigma);
}
