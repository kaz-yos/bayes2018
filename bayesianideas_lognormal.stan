/* Adopted from https://github.com/jburos/biostan/blob/master/inst/stan/weibull_survival_model.stan */
/*  Variable naming:
    obs       = observed
    cen       = (right) censored
    N         = number of samples
    M         = number of covariates
    bg        = established risk (or protective) factors
    tau       = scale parameter
*/
// Tomi Peltola, tomi.peltola@aalto.fi

functions {
    vector sqrt_vec(vector x) {
        vector[dims(x)[1]] res;

        for (m in 1:dims(x)[1]){
            res[m] = sqrt(x[m]);
        }

        return res;
    }

    vector bg_prior_lp(real r_global, vector r_local) {
        r_global ~ normal(0.0, 10.0);
        r_local ~ inv_chi_square(1.0);

        return r_global * sqrt_vec(r_local);
    }
}

data {
    int<lower=0> Nobs;
    int<lower=0> Ncen;
    int<lower=0> M_bg;
    vector[Nobs] yobs;
    vector[Ncen] ycen;
    matrix[Nobs, M_bg] Xobs_bg;
    matrix[Ncen, M_bg] Xcen_bg;
}

transformed data {
    real<lower=0> tau_mu;
    real<lower=0> tau_al;

    tau_mu = 10.0;
    tau_al = 10.0;
}

parameters {
    real<lower=0> tau_s_bg_raw;
    vector<lower=0>[M_bg] tau_bg_raw;

    real alpha_raw;
    vector[M_bg] beta_bg_raw;

    real mu;
}

transformed parameters {
    vector[M_bg] beta_bg;
    real alpha;
    real<lower=0> alpha_sq;

    beta_bg = bg_prior_lp(tau_s_bg_raw, tau_bg_raw) .* beta_bg_raw;
    /* Scale parameter */
    alpha = exp(tau_al * alpha_raw);
    alpha_sq = alpha^2;
}

model {
    /* Events contribute densities */
    yobs ~ lognormal(mu + Xobs_bg * beta_bg, alpha_sq);

    /* Censorings contribute survivals. */
    /* https://github.com/stan-dev/stan/wiki/Stan-3-Density-Notation-and-Increments */
    /* log complementary cdf. log of S = (1 - F) */
    target += lognormal_lccdf(ycen | mu + Xobs_bg * beta_bg, alpha_sq);

    beta_bg_raw ~ normal(0.0, 1.0);
    alpha_raw ~ normal(0.0, 1.0);

    mu ~ normal(0.0, tau_mu);
}

generated quantities {
    real yhat_uncens[Nobs + Ncen];
    real log_lik[Nobs + Ncen];
    real lp[Nobs + Ncen];

    for (i in 1:Nobs) {
        lp[i] = mu + Xobs_bg[i,] * beta_bg;
        yhat_uncens[i] = lognormal_rng(mu + Xobs_bg[i,] * beta_bg, alpha_sq);
        log_lik[i] = lognormal_lpdf(yobs[i] | mu + Xobs_bg[i,] * beta_bg, alpha_sq);
    }
    for (i in 1:Ncen) {
        lp[Nobs + i] = mu + Xcen_bg[i,] * beta_bg;
        yhat_uncens[Nobs + i] = lognormal_rng(mu + Xcen_bg[i,] * beta_bg, alpha_sq);
        log_lik[Nobs + i] = lognormal_lccdf(ycen[i] | mu + Xcen_bg[i,] * beta_bg, alpha_sq);
    }
}
