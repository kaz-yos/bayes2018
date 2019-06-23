data {
    // Hyperparameters
    // AR(p)
    int<lower=0> p;
    real init_mean[p];
    real<lower=0> init_sd[p];
    real theta_mean[p];
    real<lower=0> theta_sd[p];
    real<lower=0> sigma_mean;
    real<lower=0> sigma_sd;
    // Data
    int<lower=0> N;
    real y[N];
    int yr[N];
    // Forecasting
    int<lower=0> K;
}

transformed data {
}

parameters {
    real theta[p];
    real<lower=0> sigma;
}

transformed parameters {

}

model {
    // Priors
    for (i in 1:p) {
        target += normal_lpdf(theta[i] | theta_mean[i], theta_sd[i]);
    }
    target += normal_lpdf(sigma | sigma_mean, sigma_sd);
    // The first p time points need appropriate vague distributions.
    for (t in 1:p) {
        target += normal_lpdf(y[t] | init_mean[t], init_sd[t]);
    }
    // AR(p) process
    for (t in (p + 1):N) {
        real m_t = 0;
        for (i in 1:p) {
            m_t += theta[i] * y[t - i];
        }
        target += normal_lpdf(y[t] | m_t, sigma);
    }
}

generated quantities {
    real y_rep[N + K];
    for (t in 1:p) {
        y_rep[t] = normal_rng(init_mean[t], init_sd[t]);
    }
    // AR(p) process
    for (t in (p + 1):(N + K)) {
        real m_t = 0;
        for (i in 1:p) {
            m_t += theta[i] * y_rep[t - i];
        }
        y_rep[t] = normal_rng(m_t, sigma);
    }
}
