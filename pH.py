import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel
import matplotlib.pyplot as plt
import arviz as az
from stanio.json import process_dictionary

# Number of samples
num_samples = 52

num_calibration = 3
cali_meas = [3.98, 7.01, 9.01]
cali_real = [4.00, 7.00, 9.00]

# Expected pH
neutral = 7.0

# Make a list with the measured pH values
measurements = [
    6.55, 6.55, 6.96, 7.01, 
    6.92, 6.92, 6.94, 7.00, 
    6.91, 7.05, 6.98, 7, 
    6.91, 6.97, 6.92, 6.96, 
    7.00, 6.94, 6.88, 6.91, 
    6.89, 6.88, 6.92, 6.92, 
    6.92, 6.91, 6.90, 6.99, 
    6.96, 7.01, 6.9, 6.95, 
    6.89, 6.9, 6.89, 6.95, 
    6.88, 6.90, 6.93, 6.96, 
    6.94, 6.93, 6.94, 6.88, 
    6.88, 6.9, 6.86, 6.91, 
    6.88, 6.99, 6.92, 6.88
    ]

model = CmdStanModel(stan_file="pH_v2.stan")

prior_data = process_dictionary({
        "N": num_samples,
        "measurements" : measurements,
        "ave": neutral,
        "N_calibration_measurement": num_calibration,
        "calibration_measurements": cali_meas,
        "calibration_ph": cali_real
    }
)

mcmc_res = model.sample(data=prior_data)

print(mcmc_res.summary())
print(mcmc_res.diagnose())

idata = az.from_cmdstanpy(mcmc_res)

idata.to_json("my_arviz_idata.json")

true_vals = idata.posterior.true_ph.values.flatten()
sim_vals = idata.posterior.yrep.values.flatten()

w = 0.02

plt.hist(sim_vals, label="Simulated measurements", density=True, bins=np.arange(min(sim_vals), max(sim_vals) + w, w), alpha=0.5)
plt.hist(measurements, label="True measurements", density=True, bins=np.arange(min(measurements), max(measurements) + w, w), alpha=0.5)

plt.legend()
plt.show()

plt.hist(true_vals, label="True values")