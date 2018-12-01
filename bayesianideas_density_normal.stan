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
    real mu;
    // Population variance (a positive real number)
    real<lower=0> sigma_squared;
}

transformed parameters {
    // Population standard deviation (a positive real number)
    real<lower=0> sigma;
    // Standard deviation (derived from variance)
    sigma = sqrt(sigma_squared);
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

generated quantities {

    real log_f[grid_length];

    for (g in 1:grid_length) {
        // Definiting here avoids reporting of these intermediates.
        real grid_value;
        grid_value = grid_min + grid_step * (g - 1);
        log_f[g] = normal_lpdf(grid_value | mu, sigma);
    }

}
