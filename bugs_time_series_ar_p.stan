data {
    // Hyperparameters
    // AR(p)
    int <lower=0> p;
    real init_mean[p];
    real init_sd[p];
    // Data
    int<lower=0> N;
    real y[N];
    int yr[N];
}

transformed data {
}

parameters {
    real theta[p];
    real sigma;
}

transformed parameters {

}

model {
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

}
