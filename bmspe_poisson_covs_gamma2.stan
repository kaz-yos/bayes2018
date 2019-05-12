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
    vector[N] lambda;
    real<lower=0> sigma2p;
}

transformed parameters {
    vector[N] eta;
    vector[N] aa;
    vector[N] bb;

    eta = X * beta;
    aa = (exp(eta) .* exp(eta)) / sigma2p;
    bb = exp(eta) / sigma2p;
}

model {
    /* Prior */
    for (j in 1:M) {
        target += normal_lpdf(beta[j] | 0, s);
    }
    target += inv_gamma_lpdf(sigma2p | a, b);

    /* Likelihood */
    /* y_i ~ poisson(exp(eta_i)); */
    for (i in 1:N) {
        target += gamma_lpdf(lambda[i] | aa[i], bb[i]);
        target += poisson_lpmf(y[i] | lambda[i]);
    }
}

generated quantities {
    int y_new[N];
    for (i in 1:N) {
        if (lambda[i] < -20) {
            /* To avoid erros like the below during the warmup. */
            /* Check posterior predictive. */
            y_new[i] = -1;
        } else {
            y_new[i] = poisson_rng(lambda[i]);
        }
    }
}
