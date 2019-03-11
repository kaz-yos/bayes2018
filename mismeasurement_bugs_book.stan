data {
    /* N: Number of rows */
    int<lower=0> N;
    int<lower=0,upper=1> Y[N];
    int<lower=-1,upper=1> X_true[N];
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
    /* psi = P(X_true = 1) */
    real<lower=0,upper=1> psi;
}

transformed parameters {

}

model {
    /* individual-level lp at X_true = 0 and X_true = 1 */
    matrix[N,2] lp;

    /* Priors */
    /*  Outcome model parameters */
    target += normal_lpdf(beta0 | 0, 100);
    target += normal_lpdf(beta1 | 0, 100);
    /*  Error model parameter */
    target += uniform_lpdf(phi[1,1] | 0, 1);
    target += uniform_lpdf(phi[1,2] | 0, 1);
    target += uniform_lpdf(phi[2,1] | 0, 1);
    target += uniform_lpdf(phi[2,2] | 0, 1);
    /*  Covariate model parameter */
    target += uniform_lpdf(psi | 0, 1);

    /* Loop over rows */
    for (i in 1:N) {
        if (R_X[i] == 1) {
            /* Contribution for a row with OBSERVED X */
            /* count[i] to account for the sample size */
            target += count[i] *
                (/*  Outcome model */
                 bernoulli_lpmf(Y[i] | inv_logit(beta0 + beta1 * X_true[i]))
                 /*  Error model */
                 + bernoulli_lpmf(X_mis[i] | phi[X_true[i]+1, Y[i]+1])
                 /*  Covariate model */
                 + bernoulli_lpmf(X_true[i] | psi));

        } else {
            /* Contribution for a row with UNOBSERVED X_true (marginalized over X_true) */

            /* X_true = 0 type contribution */
            /* p(Yi|Xi=0,beta) p(Xi*|Xi=0,phi) p(Xi=0|psi) */
            lp[i,1] = (/* Outcome model */
                       bernoulli_lpmf(Y[i] | inv_logit(beta0))
                       /* Error model */
                       + bernoulli_lpmf(X_mis[i] | phi[1, Y[i]+1])
                       /* Covariate model */
                       + log(1-psi));

            /* X_true = 1 type contribution */
            /* p(Yi|Xi=1,beta)p(Xi*|Xi=1,phi)p(Xi=1|psi) */
            lp[i,2] = (/* Outcome model */
                       bernoulli_lpmf(Y[i] | inv_logit(beta0 + beta1))
                       /* Error model */
                       + bernoulli_lpmf(X_mis[i] | phi[2, Y[i]+1])
                       /* Covariate model */
                       + log(psi));

            /* Sum up using log_sum_exp to marginalize over X_true */
            /* count[i] to account for the sample size */
            target += count[i] * log_sum_exp(lp[i]);
        }
    }

}

generated quantities {
    real p_X[N];

    for (i in 1:N) {
        if (R_X[i] == 1) {
            /* Observed X_true if available */
            p_X[i] = X_true[i];
        } else {
            p_X[i] = exp(lp[i,1] - log_sum_exp(lp[i]));
        }
    }
}
