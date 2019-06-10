data {
    // Hyperparameter for lambda[k]
    real<lower=0> w;
    real<lower=0> lambda_star;
    // Hyperparameter for beta
    real beta_mean;
    real<lower=0> beta_sd;
    // Number of pieces
    int<lower=0> K;
    real cutpoints[K];
    //
    int<lower=0> N;
    int<lower=0,upper=1> cens[N];
    real y[N];
    int<lower=0,upper=1> x[N];
}

transformed data {
}

parameters {
    // Baseline hazards
    real lambda[K];
    // Effect of group
    real beta;
}

transformed parameters {
}

model {
    // Prior on beta
    target += normal_lpdf(beta | beta_mean, beta_sd);

    // Loop over pieces of time
    for (k in 1:(K - 1)) {
        // start/end of interval
        real start = cutpoints[k];
        real end = cutpoints[k+1];
        real length = cutpoints[k+1] - cutpoints[k];

        // Prior on lambda
        // BIDA 13.2.5 Priors for lambda
        if (k == 1) {
            // The first interval requires special handling.
            target += gamma_lpdf(lambda[1] | 10000 * 0.01, 10000);
        } else {
            // Mean lambda_star
            target += gamma_lpdf(lambda[k] | lambda_star * length * w, length * w);
        }

        // Likelihood contribution
        // BIDA 13.2.3 Likelihood for piecewise hazard PH model
        for (i in 1:N) {
            // Linear predictor
            real lp = beta * x;
            // Everyone will contribute to survival part
            if (y[i] >= cutpoints[k+1]) {
                // If surviving beyond end of interval,
                // contribute survival throughout interval.
                target += -exp(lp) * (lambda[k] * length);
                //
            } else if (cutpoints[k] <= y[i] & y[i] < cutpoints[k+1]) {
                // If ending follow up during the interval,
                // Contribute survival until end of follow up.
                target += -exp(lp) * (lambda[k] * (y[i] - cutpoints[k]));
                //
                // Event individuals also contribute to hazard part
                if (cens[i] == 1) {
                  target += -exp(lp) * lambda[k];
                }
            } else {
                // If having ended follow up,
                // No contribution in this interval
            }
        }
    }
}

generated quantities {

}
