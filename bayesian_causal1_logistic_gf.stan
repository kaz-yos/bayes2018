data {
    // Number of parameters including intercept
    int<lower=1> p;
    // Hyperparameters
    real beta_mean[p];
    real<lower=0> beta_sd[p];
    // Number of observations
    int<lower=1> N;
    // Binary outcome
    int<lower=0,upper=1> y[N];
    // Model matrix
    matrix[N,p] X;
    // Counterfactual model matrix
    int<lower=1> N_new;
    matrix[N_new,p] X0;
    matrix[N_new,p] X1;
    // Whether to evaluate likelihood
    int<lower=0,upper=1> use_lik;
}

transformed data {

}

parameters {
    // Real vector of p dimension.
    vector[p] beta;
}

transformed parameters {
    // Linear predictor
    vector[N] eta = X * beta;
}

model {
    // Prior
    for (j in 1:p) {
        target += normal_lpdf(beta[j] | beta_mean[j], beta_sd[j]);
    }

    // Likelihood for logistic model
    if (use_lik == 1) {
        for (i in 1:N) {
            // https://mc-stan.org/docs/2_19/functions-reference/bernoulli-logit-distribution.html
            target += bernoulli_logit_lpmf(y[i] | eta[i]);
        }
    }
}

generated quantities {
    // Counterfactual probability of outcomes
    vector[N_new] pY0 = inv_logit(X0 * beta);
    vector[N_new] pY1 = inv_logit(X1 * beta);
    // Counterfactual risk difference
    real rd = mean(pY1) - mean(pY0);
    // Counterfactual risk ratio
    real<lower=0> rr = mean(pY1) / mean(pY0);

    // Other elements
    vector[N] log_lik;
    int<lower=0,upper=1> y_rep[N];

    for (i in 1:N) {
        // Observation level log likelihood
        log_lik[i] = bernoulli_logit_lpmf(y[i] | eta[i]);
        // Predicted (note these are prediction wrt observed assignment)
        y_rep[i] = bernoulli_logit_rng(eta[i]);
    }
}
