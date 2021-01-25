import pandas as pd

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.contrib.control_flow import scan
from numpyro.diagnostics import autocorrelation, hpdi
from numpyro.infer import MCMC, NUTS, Predictive


def load_dataset() -> pd.DataFrame:

    URL = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/lynx.csv"
    df = pd.read_csv(URL, index_col=0)

    return df



