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
    // Whether to use likelihood
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
    // http://modernstatisticalworkflow.blogspot.com/2017/04/an-easy-way-to-simulate-fake-data-from.html?m=1
    if (use_lik == 1) {
        for (i in 1:N) {
            // https://mc-stan.org/docs/2_19/functions-reference/bernoulli-logit-distribution.html
            target += bernoulli_logit_lpmf(y[i] | eta[i]);
        }
    }
}

generated quantities {

}
