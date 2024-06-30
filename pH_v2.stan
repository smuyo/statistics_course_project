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
    int<lower=1> N_calibration_measurement;
    vector[N] measurements;
    vector[N_calibration_measurement] calibration_measurements;
    vector[N_calibration_measurement] calibration_ph;
    real ave;
}

transformed data {
    vector[N] meas_ls = standardise(log(measurements), mean(log(measurements)), sd(log(measurements)));
    real ave_ls = standardise(log(ave), mean(log(measurements)), sd(log(measurements)));
}

parameters {
    real true_ph_ls;
    real<lower=0> biological_noise;
    real bias;
    real curr_err;
}

model {
    curr_err ~ lognormal(log(0.01), 0.1);
    biological_noise ~ lognormal(log(0.05), 0.1);
    meas_ls ~ normal(true_ph_ls + bias, curr_err + biological_noise);
    calibration_measurements ~ normal(calibration_ph + bias, curr_err);
}

generated quantities {
    real true_ph = exp(unstandardise(true_ph_ls, mean(log(measurements)), sd(log(measurements))));
    real yrep = exp(unstandardise(normal_rng(true_ph_ls + bias, curr_err + biological_noise), mean(log(measurements)), sd(log(measurements))));
}