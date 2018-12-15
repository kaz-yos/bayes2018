data {
    /* Hyperparameters*/
    real<lower=0> a;
    real<lower=0> b;

    /* Dimensions */
    int<lower=0> N;
    int<lower=0> M;
    /* Design Matrix */
    matrix[N,M] X;
    /* Outcome (a real vector of length n) */
    int<lower=0> y[N];
}

parameters {
    real<lower=0> lambda;
}

model {
    /* Prior */
    /* lambda ~ gamma(a, b); */
    /* Explicit contribution to target */
    target += gamma_lpdf(lambda | a, b);

    /* Likelihood */
    /* y ~ poisson(lambda); */
    /* Explicit contribution to target */
    for (i in 1:N) {
        target += poisson_lpmf(y[i] | lambda);
    }
}

generated quantities {
    int<lower=0> y_new[N];
    for (i in 1:N) {
        y_new[i] = poisson_rng(lambda);
    }
}
