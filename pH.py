import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel
import matplotlib.pyplot as plt
import arviz as az
from stanio.json import process_dictionary

# Number of samples
num_samples = 52

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

# Decare the error of the pH meter (from the web)
meter_error = 0.01

model = CmdStanModel(stan_file="pH.stan")

prior_data = process_dictionary({
        "N": num_samples,
        "measurements" : measurements,
        "ave": neutral,
        "err": meter_error,
        "num_sims": 100
    }
)

mcmc_res = model.sample(data=prior_data)

print(mcmc_res.summary())
print(mcmc_res.diagnose())

idata = az.from_cmdstanpy(mcmc_res)

print(idata)
print(idata.sample_stats)
print(idata.posterior)

idata.to_json("my_arviz_idata.json")

az.plot_posterior(
    idata.posterior.ph_vals,
    kind="hist",
    hdi_prob="hide",
    ref_val=[6.88,7.01]
)

plt.show()
