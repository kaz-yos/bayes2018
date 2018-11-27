data {
    // Hyperparameters
    real alpha;
    real beta;
    real m;
    real<lower=0> s_squared;
    real<lower=0> dirichlet_alpha;

    // Define variables in data
    // Number of observations (an integer)
    int<lower=0> n;
    // Outcome (a real vector of length n)
    real y[n];
    // Number of latent clusters
    int<lower=1> H;
}

transformed data {
    real s;
    s = sqrt(s_squared);
}

parameters {
    // Define parameters to estimate
    // Population mean (a real number)
    ordered[H] mu;
    // Population variance (a positive real number)
    real<lower=0> tau[H];
    // Cluster probability
    simplex[H] Pi;
}

transformed parameters {
    // Population standard deviation (a positive real number)
    real<lower=0> sigma[H];
    // Standard deviation (derived from variance)
    sigma = sqrt(1 / tau);
}

model {
    // Temporary vector for loop use. Need to come first before priors.
    real contributions[H];

    // Prior part of Bayesian inference
    // All vectorized
    // Mean
    mu ~ normal(m, s);
    // tau = 1/sigma^2 has gamma prior
    tau ~ gamma(alpha, beta);
    // cluster probability vector
    Pi ~ dirichlet(rep_vector(dirichlet_alpha / H, H));

    // Likelihood part of Bayesian inference
    // Outcome model N(mu, sigma^2) (use SD rather than Var)
    for (i in 1:n) {
        // Loop over individuals
        // z[i] in {1,...,H} gives the cluster membership.
        /* y[i] ~ normal(mu[z[i]], sigma[z[i]]); */

          for (h in 1:H) {
              // Loop over clusters within each individual
              // Log likelihood contributions log(Pi[h] * N(y[i] | mu[h],sigma[h]))
              contributions[h] = log(Pi[h]) + normal_lpdf(y[i] | mu[h], sigma[h]);
          }

          // log(sum(exp(contribution element)))
          target += log_sum_exp(contributions);

    }
}
