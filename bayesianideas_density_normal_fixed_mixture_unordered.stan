data {
    // Hyperparameters
    real alpha;
    real beta;
    real m;
    real s_squared;
    real<lower=0> dirichlet_alpha;

    // Define variables in data
    // Number of observations (an integer)
    int<lower=0> n;
    // Outcome (a real vector of length n)
    real y[n];
    // Number of latent clusters
    int<lower=1> H;

    // Grid evaluation
    real grid_max;
    real grid_min;
    int<lower=1> grid_length;
}

transformed data {
    real s;
    real grid_step;

    s = sqrt(s_squared);
    grid_step = (grid_max - grid_min) / (grid_length - 1);
}

parameters {
    // Define parameters to estimate
    // Population mean (a real number)
    vector[H] mu;
    // Population variance (a positive real number)
    real<lower=0> sigma_squared[H];
    // Cluster probability
    simplex[H] Pi;
}

transformed parameters {
    // Population standard deviation (a positive real number)
    real<lower=0> sigma[H];
    // Standard deviation (derived from variance)
    sigma = sqrt(sigma_squared);
}

model {
    // Temporary vector for loop use. Need to come first before priors.
    real contributions[H];

    // Prior part of Bayesian inference
    // All vectorized
    // Mean
    mu ~ normal(m, s);
    // sigma^2 has inverse gamma (alpha = 1, beta = 1) prior
    sigma_squared ~ inv_gamma(alpha, beta);
    // cluster probability vector
    Pi ~ dirichlet(rep_vector(dirichlet_alpha / H, H));

    // Likelihood part of Bayesian inference
    // Outcome model N(mu, sigma^2) (use SD rather than Var)
    for (i in 1:n) {
        // Loop over individuals

          for (h in 1:H) {
              // Loop over clusters within each individual
              // Log likelihood contributions log(Pi[h] * N(y[i] | mu[h],sigma[h]))
              contributions[h] = log(Pi[h]) + normal_lpdf(y[i] | mu[h], sigma[h]);
          }

          // log(sum(exp(contribution element)))
          target += log_sum_exp(contributions);

    }
}

generated quantities {

    real log_f[grid_length];

    for (g in 1:grid_length) {
        // Definiting here avoids reporting of these intermediates.
        real contributions[H];
        real grid_value;

        grid_value = grid_min + grid_step * (g - 1);
        for (h in 1:H) {
            contributions[h] = log(Pi[h]) + normal_lpdf(grid_value | mu[h], sigma[h]);
        }

        log_f[g] = log_sum_exp(contributions);
    }

}
