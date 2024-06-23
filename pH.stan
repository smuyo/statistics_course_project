functions {
  vector standardise(vector v, real m, real s) {
    return (v - m) / s;
  }
  real standardise(real v, real m, real s) {
    return (v - m) / s;
  }
  vector unstandardise(vector u, real m, real s) {
    return m + u * s;
  }
  real unstandardise(real u, real m, real s) {
    return m + u * s;
  }
}

data {
    int<lower=1> N;
    vector[N] measurements;
    int<lower=1> num_sims;
    real ave;
    real err;
}

transformed data {
    vector[N] meas_ls = standardise(log(measurements), mean(log(measurements)), sd(log(measurements)));
    real ave_ls = standardise(log(ave), mean(log(measurements)), sd(log(measurements)));
}

parameters {
    real bias;
    real curr_err;
    real sim_ph;
}

model {
    bias ~ lognormal(0,0.1);
    curr_err ~ normal(0,err);
    sim_ph ~ normal(ave_ls + curr_err, bias);
}

generated quantities {
    real ph_vals = exp(unstandardise(sim_ph, mean(log(measurements)), sd(log(measurements))));
}