data {
    /* Outcome data */
    int<lower=0> N;
    real<lower=0,upper=1> y[N];
    /* X1 matrix for the mean parameter. Include a constant column. */
    int<lower=1> X1_dim;
    matrix[N, X1_dim] X1;
    real beta_x1_mean[X1_dim];
    real<lower=0> beta_x1_sd[X1_dim];
    /* X2 matrix for the precision parameter. Include a constant column. */
    int<lower=1> X2_dim;
    matrix[N, X2_dim] X2;
    real beta_x2_mean[X2_dim];
    real<lower=0> beta_x2_sd[X2_dim];
}

transformed data {

}

parameters {
    vector[X1_dim] beta_x1;
    vector[X2_dim] beta_x2;
}

transformed parameters {
    vector[N] eta_x1 = X1 * beta_x1;
    vector[N] eta_x2 = X2 * beta_x2;

    /* logit for mean. expit is the inverse. */
    vector[N] mu = inv_logit(eta_x1);
    /* log for precision. exp is the inverse. */
    vector[N] phi = exp(eta_x2);
}

model {

    /* Priors */
    for (j in 1:X1_dim) {
        target += normal_lpdf(beta_x1[j] | beta_x1_mean[j], beta_x1_sd[j]);
    }
    for (k in 1:X2_dim) {
        target += normal_lpdf(beta_x2[k] | beta_x2_mean[k], beta_x2_sd[k]);
    }

    /* Mean model */
    for (i in 1:N) {
        target += beta_lpdf(y[i] | (mu .* phi)[i], ((1-mu) .* phi)[i]);
    }

}

generated quantities {
    /* Posterior predictive */
    real<lower=0,upper=1> y_rep[N];

    for (i in 1:N) {
        y_rep[i] = beta_rng((mu .* phi)[i], ((1-mu) .* phi)[i]);
    }
}
