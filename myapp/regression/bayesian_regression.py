from typing import Any, Callable, Optional

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from jax import random, vmap
from jax.scipy.special import logsumexp
from numpyro import handlers
from numpyro.diagnostics import hpdi
from numpyro.infer import MCMC, NUTS, Predictive


def load_dataset() -> pd.DataFrame:

    DATASET_URL = (
        "https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/WaffleDivorce.csv"
    )
    df = pd.read_csv(DATASET_URL, sep=";")

    def standardize(x: np.ndarray) -> np.ndarray:
        return (x - x.mean()) / x.std()

    df["AgeScaled"] = df["MedianAgeMarriage"].pipe(standardize)
    df["MarriageScaled"] = df["Marriage"].pipe(standardize)
    df["DivorceScaled"] = df["Divorce"].pipe(standardize)

    return df


def model(
    marriage: Optional[np.ndarray] = None,
    age: Optional[np.ndarray] = None,
    divorce: Optional[np.ndarray] = None,
) -> None:

    a = numpyro.sample("a", dist.Normal(0.0, 0.2))

    if marriage is not None:
        bM = numpyro.sample("bM", dist.Normal(0.0, 0.5))
        M = bM * marriage
    else:
        M = 0

    if age is not None:
        bA = numpyro.sample("bA", dist.Normal(0.0, 0.5))
        A = bA * age
    else:
        A = 0

    sigma = numpyro.sample("sigma", dist.Exponential(1.0))
    mu = a + M + A
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=divorce)


def predict(
    rng_key: np.ndarray, post_samples: np.ndarray, model: Callable, *args: Any, **kwargs: Any
) -> np.ndarray:

    model = handlers.seed(handlers.condition(model, post_samples), rng_key)
    model_trace = handlers.trace(model).get_trace(*args, **kwargs)
    return model_trace["obs"]["value"]


def log_likelihood(
    rng_key: np.ndarray, params: np.ndarray, model: Callable, *args: Any, **kwargs: Any
) -> np.ndarray:

    model = handlers.condition(model, params)
    model_trace = handlers.trace(model).get_trace(*args, **kwargs)
    obs_node = model_trace["obs"]
    return obs_node["fn"].log_prob(obs_node["value"])


def log_pred_density(
    rng_key: np.ndarray, params: jnp.ndarray, model: Callable, *args: Any, **kwargs: Any
) -> np.ndarray:

    n = list(params.values())[0].shape[0]
    log_lk_fn = vmap(
        lambda rng_key, params: log_likelihood(rng_key, params, model, *args, **kwargs)
    )
    log_lk_vals = log_lk_fn(random.split(rng_key, n), params)
    return (logsumexp(log_lk_vals, 0) - jnp.log(n)).sum()


def main() -> None:

    df = load_dataset()

    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)

    # Inference posterior
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=2000)
    mcmc.run(rng_key_, marriage=df["MarriageScaled"].values, divorce=df["DivorceScaled"].values)
    mcmc.print_summary()
    samples_1 = mcmc.get_samples()

    # Compute empirical posterior distribution
    posterior_mu = (
        jnp.expand_dims(samples_1["a"], -1)
        + jnp.expand_dims(samples_1["bM"], -1) * df["MarriageScaled"].values
    )

    mean_mu = jnp.mean(posterior_mu, axis=0)
    hpdi_mu = hpdi(posterior_mu, 0.9)
    print(mean_mu, hpdi_mu)

    # Posterior predictive distribution
    rng_key, rng_key_ = random.split(rng_key)
    predictive = Predictive(model, samples_1)
    predictions = predictive(rng_key_, marriage=df["MarriageScaled"].values)["obs"]
    df["MeanPredictions"] = jnp.mean(predictions, axis=0)
    print(df.head())

    # Predictive utility with effect handlers
    predict_fn = vmap(
        lambda rng_key, samples: predict(
            rng_key, samples, model, marriage=df["MarriageScaled"].values
        )
    )
    predictions_1 = predict_fn(random.split(rng_key_, 2000), samples_1)
    mean_pred = jnp.mean(predictions_1, axis=0)
    print(mean_pred)

    # Posterior predictive density
    rng_key, rng_key_ = random.split(rng_key)
    lpp_dns = log_pred_density(
        rng_key_,
        samples_1,
        model,
        marriage=df["MarriageScaled"].values,
        divorce=df["DivorceScaled"].values,
    )
    print("Log posterior predictive density", lpp_dns)


if __name__ == "__main__":
    main()
