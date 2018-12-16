data {
    /* Hyperparameters*/
    real<lower=0> a;
    real<lower=0> b;
    real<lower=0> s;

    /* Dimensions */
    int<lower=0> N;
    int<lower=0> M;
    /* Design Matrix */
    matrix[N,M] X;
    /* Outcome */
    int<lower=0> y[N];
}

parameters {
    vector[M] beta;
    vector<lower=0>[N] gamma;
    real<lower=0> aa;
}

transformed parameters {
    vector[N] eta;
    vector<lower=0>[N] mu;

    eta = X * beta;
    mu = exp(eta);
}

model {
    /* Prior */
    for (j in 1:M) {
        /* beta_j ~ N(0, s) */
        target += normal_lpdf(beta[j] | 0, s);
    }
    /* aa ~ Gamma(a, b) */
    target += gamma_lpdf(aa | a, b);

    /* Likelihood */
    for (i in 1:N) {
        /* gamma_i ~ Gamma(aa, bb) */
        /* E[gamma_i] = 1 must be met for identifiability. */
        target += gamma_lpdf(gamma[i] | 1/aa, 1/aa);
        /* y_i ~ poisson(gamma_i * mu_i); */
        target += poisson_lpmf(y[i] | gamma[i] * mu[i]);
    }
}

generated quantities {
    int y_new[N];
    for (i in 1:N) {
        if (gamma[i] * mu[i] > 1e+09) {
            /* To avoid erros like the below during the warmup. */
            /* Exception: poisson_rng: Rate parameter is 1.92222e+40, but must be less than 1.07374e+09 */
            /* Check posterior predictive. */
            y_new[i] = -1;
        } else {
            y_new[i] = poisson_rng(gamma[i] * mu[i]);
        }
    }
}
