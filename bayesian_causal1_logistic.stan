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
    for (i in 1:N) {
        // https://mc-stan.org/docs/2_19/functions-reference/bernoulli-logit-distribution.html
        target += bernoulli_logit_lpmf(y[i] | eta[i]);
    }
}

generated quantities {

}
