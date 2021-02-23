from typing import Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from jax import random
from numpyro.contrib.control_flow import scan
from numpyro.diagnostics import hpdi
from numpyro.infer import MCMC, NUTS, Predictive


def load_dataset() -> pd.DataFrame:

    URL = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/lynx.csv"  # noqa
    df = pd.read_csv(URL, index_col=0)

    return df.reset_index(drop=True)


def sgt(y: jnp.ndarray, seasonality: int, future: int = 0) -> None:

    cauchy_sd = jnp.max(y) / 150

    nu = numpyro.sample("nu", dist.Uniform(2, 20))
    powx = numpyro.sample("powx", dist.Uniform(0, 1))
    sigma = numpyro.sample("sigma", dist.HalfCauchy(cauchy_sd))
    offset_sigma = numpyro.sample(
        "offset_sigma", dist.TruncatedCauchy(low=1e-10, loc=1e-10, scale=cauchy_sd)
    )

    coef_trend = numpyro.sample("coef_trend", dist.Cauchy(0, cauchy_sd))
    pow_trend_beta = numpyro.sample("pow_trend_beta", dist.Beta(1, 1))
    pow_trend = 1.5 * pow_trend_beta - 0.5
    pow_season = numpyro.sample("pow_season", dist.Beta(1, 1))

    level_sm = numpyro.sample("level_sm", dist.Beta(1, 2))
    s_sm = numpyro.sample("s_sm", dist.Uniform(0, 1))
    init_s = numpyro.sample("init_s", dist.Cauchy(0, y[:seasonality] * 0.3))

    num_lim = y.shape[0]

    def transition_fn(
        carry: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], t: jnp.ndarray
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:

        level, s, moving_sum = carry
        season = s[0] * level ** pow_season
        exp_val = level + coef_trend * level ** pow_trend + season
        exp_val = jnp.clip(exp_val, a_min=0)
        y_t = jnp.where(t >= num_lim, exp_val, y[t])

        moving_sum = moving_sum + y[t] - jnp.where(t >= seasonality, y[t - seasonality], 0.0)
        level_p = jnp.where(t >= seasonality, moving_sum / seasonality, y_t - season)
        level = level_sm * level_p + (1 - level_sm) * level
        level = jnp.clip(level, a_min=0)

        new_s = (s_sm * (y_t - level) / season + (1 - s_sm)) * s[0]
        new_s = jnp.where(t >= num_lim, s[0], new_s)
        s = jnp.concatenate([s[1:], new_s[None]], axis=0)

        omega = sigma * exp_val ** powx + offset_sigma
        y_ = numpyro.sample("y", dist.StudentT(nu, exp_val, omega))

        return (level, s, moving_sum), y_

    level_init = y[0]
    s_init = jnp.concatenate([init_s[1:], init_s[:1]], axis=0)
    moving_sum = level_init
    with numpyro.handlers.condition(data={"y": y[1:]}):
        _, ys = scan(
            transition_fn, (level_init, s_init, moving_sum), jnp.arange(1, num_lim + future)
        )

    if future > 0:
        numpyro.deterministic("y_forecast", ys[-future:])


def plot_results(
    time_series: np.ndarray, y: np.ndarray, forecast_marginal: jnp.ndarray, test_index: int
) -> None:

    t_test = time_series[test_index:]
    y_test = y[test_index:]

    y_pred = jnp.mean(forecast_marginal, axis=0)
    smape = jnp.mean(jnp.abs(y_pred - y_test) / (y_pred + y_test)) * 200
    msqrt = jnp.sqrt(jnp.mean((y_pred - y_test) ** 2))
    hpd_low, hpd_high = hpdi(forecast_marginal)

    plt.figure(figsize=(8, 4))
    plt.plot(time_series, y)
    plt.plot(t_test, y_pred, lw=2)
    plt.fill_between(t_test, hpd_low, hpd_high, alpha=0.3)
    plt.title(f"Forecasting lynx dataset with SGT (sMAPE: {smape:.2f}, RMSE: {msqrt:.2f})")
    plt.tight_layout()
    plt.show()


def main() -> None:

    numpyro.set_host_device_count(4)

    df = load_dataset()
    test_index = 80
    test_len = len(df) - test_index
    y_train = jnp.array(df.loc[:test_index, "value"], dtype=jnp.float32)

    # Inference
    kernel = NUTS(sgt)
    mcmc = MCMC(kernel, num_warmup=500, num_samples=500, num_chains=4)
    mcmc.run(random.PRNGKey(0), y_train, seasonality=38)
    mcmc.print_summary()
    samples = mcmc.get_samples()

    # Prediction
    predictive = Predictive(sgt, samples, return_sites=["y_forecast"])
    prediction_sample = predictive(random.PRNGKey(1), y_train, seasonality=38, future=test_len)
    forecast_marginal = prediction_sample["y_forecast"]

    plot_results(df["time"].values, df["value"].values, forecast_marginal, test_index=test_index)


if __name__ == "__main__":
    main()
