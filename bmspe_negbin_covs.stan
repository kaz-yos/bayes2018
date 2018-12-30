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
    real<lower=0> a_gamma;
}

transformed parameters {
    vector[N] eta;
    real<lower=0> phi;

    eta = X * beta;
    phi = 1 / a_gamma;
}

model {
    /* Prior */
    for (j in 1:M) {
        target += normal_lpdf(beta[j] | 0, s);
    }
    target += gamma_lpdf(a_gamma | a, b);

    /* Likelihood */
    /* y_i ~ poisson(exp(eta_i)); */
    for (i in 1:N) {
        /* https://mc-stan.org/docs/2_18/functions-reference/nbalt.html */
        target += neg_binomial_2_lpmf(y[i] | exp(eta[i]), phi);
    }
}

generated quantities {
    int y_new[N];
    vector[N] log_lik;

    for (i in 1:N) {
        /* eta[i] is the log(mean). */
        if (eta[i] > 15) {
            /* To avoid erros like the below during the warmup. */
            /* neg_binomial_2_rng: Random number that came from gamma distribution is 3.02668e+39, but must be less than 1.07374e+09 */
            /* https://groups.google.com/forum/#!topic/stan-users/4g2hbwtRELQ */
            /* Check posterior predictive for anomaly. */
            y_new[i] = -1;
        } else {
            y_new[i] = neg_binomial_2_rng(exp(eta[i]), phi);
        }

        log_lik[i] = neg_binomial_2_lpmf(y[i] | exp(eta[i]), phi);
    }
}
