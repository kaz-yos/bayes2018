data {
    real<lower=0> alpha;
    real<lower=0> beta;

    /* N: Number of rows */
    int<lower=0> N;
    int<lower=0,upper=1> Y[N];
    int<lower=0,upper=1> X_true[N];
    int<lower=0,upper=1> X_mis[N];
    int<lower=0> count[N];
    int<lower=0,upper=1> R_X[N];
}

transformed data {

}

parameters {
    /* Outcome model parameters */
    real beta0;
    real beta1;
    /* Error model parameter */
    real<lower=0,upper=1> phi[2,2];
    /* phi[1,1] = P(X_mis = 1 | X_true = 0, Y = 0) */
    /* phi[1,2] = P(X_mis = 1 | X_true = 0, Y = 1) */
    /* phi[2,1] = P(X_mis = 1 | X_true = 1, Y = 0) */
    /* phi[2,2] = P(X_mis = 1 | X_true = 1, Y = 1) */
    /* Covariate model parameter */
    /* pwi = P(X_true = 1) */
    real<lower=0,upper=1> psi;
}

transformed parameters {

}

model {

    /* Priors */

    /* Loop over rows */
    for (i in 1:N) {
        if (R_X[i] == 1) {
            /* Contribution for a row with observed X */
            /*  Outcome model */
            /*   count[i] to account for the sample size */
            target += bernoulli_lpmf(Y[i] | inv_logit(beta0 + beta1 * X[i])) * count[i];
            /*  Error model */
            /*   count[i] to account for the sample size */
            target += bernoulli_lpmf(X_mis[i] | phi[X_true[i]+1, Y[i]+1]) * count[i];
            /*  Covariate model */
            target += bernoulli_lpmf(X_true[i] | psi) * count[i];

        } else {
            /* Contribution for a row with unobserved X */

        }
    }

}

generated quantities {

}
