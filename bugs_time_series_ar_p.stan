data {
    // Hyperparameters
    // AR(p)
    int<lower=0> p;
    real<lower=0> epsilon_sd[p];
    real theta0_mean;
    real<lower=0> theta0_sd;
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
    real theta0;
    real theta[p];
    real<lower=0> sigma;
    real epsilon[p];
}

transformed parameters {
    // Mean function
    real m[N];
    // Define the first p elements through errors.
    // Then define priors for epsilon. Data do not inform these.
    for (t in 1:p) {
        m[t] = y[t] - epsilon[t];
    }
    // Define the rest through AR(p)
    for (t in (p + 1):N) {
        m[t] = theta0;
        for (i in 1:p) {
            m[t] += theta[i] * y[t - i];
        }
    }
}

model {
    // Priors
    // theta0
    target += normal_lpdf(theta0 | theta0_mean, theta0_sd);
    // theta array
    for (i in 1:p) {
        target += normal_lpdf(theta[i] | theta_mean[i], theta_sd[i]);
    }
    // sigma
    target += normal_lpdf(sigma | sigma_mean, sigma_sd);
    // The first p error terms need appropriate prior distributions.
    for (t in 1:p) {
        target += normal_lpdf(epsilon[t] | 0, epsilon_sd[t]);
    }

    // Likelihood of AR(p) process
    // The first p y values do not contribute.
    for (t in (p + 1):N) {
        target += normal_lpdf(y[t] | m[t], sigma);
    }
}

generated quantities {
    // Use all N observed y's for these.
    real m_rep[N + K];
    real y_rep[N + K];
    // Use only first p observed y's for these.
    real m_new[N + K];
    real y_new[N + K];
    // The first p y's are not modeled, so just set to what they are.
    for (t in 1:p) {
        m_rep[t] = y[t];
        y_rep[t] = y[t];
        m_new[t] = y[t];
        y_new[t] = y[t];
    }
    // AR(p) prediction based on observed y's and
    for (t in (p + 1):(N + K)) {
        m_rep[t] = theta0;
        m_new[t] = theta0;
        for (i in 1:p) {
            if (t - i <= N) {
                // Calculate m_rep using observed y's as long as they are available.
                m_rep[t] += theta[i] * y[t - i];
            } else {
                // Once it's out of bound used the generated values.
                m_rep[t] += theta[i] * y_rep[t - i];
            }
            // Calculate m_new using generated y's except for the first p.
            m_new[t] += theta[i] * y_new[t - i];
        }
        // Randomly generate y_rep[t] based on the calculated m_rep[t].
        y_rep[t] = normal_rng(m_rep[t], sigma);
        // Randomly generate y_new[t] based on the calculated m_new[t].
        y_new[t] = normal_rng(m_new[t], sigma);
    }
}
