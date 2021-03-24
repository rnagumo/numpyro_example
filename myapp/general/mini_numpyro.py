"""Mini-numpyro

http://pyro.ai/examples/minipyro.html
"""

from collections import OrderedDict
from typing import Any, Callable, Generator, Dict, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
from jax import random, vmap


NUMPYRO_STACK = []
PARAM_STORE = {}


def get_param_store():
    return PARAM_STORE


def apply_stack(msg: Dict[str, Any]) -> Dict[str, Any]:
    # https://github.com/pyro-ppl/numpyro/blob/master/numpyro/primitives.py#L21

    for pointer, handler in enumerate(reversed(NUMPYRO_STACK)):
        handler.process_message(msg)
        if msg.get("stop"):
            break

    if msg["value"] is None:
        msg["value"] = msg["fn"](*msg["args"])

    for handler in NUMPYRO_STACK[-pointer - 1:]:
        handler.postprocess_message(msg)

    return msg


class Messanger:
    # https://github.com/pyro-ppl/numpyro/blob/master/numpyro/primitives.py#L45
    def __init__(self, fn: Optional[Callable] = None) -> None:
        self.fn = fn

    def __enter__(self) -> None:
        NUMPYRO_STACK.append(self)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        assert NUMPYRO_STACK[-1] is self
        NUMPYRO_STACK.pop()

    def process_message(self, msg: Dict[str, Any]) -> None:
        pass

    def postprocess_message(self, msg: Dict[str, Any]) -> None:
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        with self:
            return self.fn(*args, **kwargs)


class trace(Messanger):
    # https://github.com/pyro-ppl/numpyro/blob/master/numpyro/handlers.py#L110
    def __enter__(self) -> OrderedDict:
        super().__enter__()
        self.trace = OrderedDict()
        return self.trace

    def postprocess_message(self, msg: Dict[str, Any]) -> None:
        assert msg["type"] != "sample" or msg["name"] not in self.trace
        self.trace[msg["name"]] = msg.copy()

    def get_trace(self, *args: Any, **kwargs: Any) -> OrderedDict:
        self(*args, **kwargs)
        return self.trace


class replay(Messanger):
    # https://github.com/pyro-ppl/numpyro/blob/master/numpyro/handlers.py#L166
    def __init__(self, fn: Callable, guide_trace: OrderedDict) -> None:
        self.guide_trace = guide_trace
        super().__init__(fn=fn)

    def process_message(self, msg: Dict[str, Any]) -> None:
        if msg["type"] in ("sample", "plate") and msg["name"] in self.guide_trace:
            msg["value"] = self.guide_trace[[msg["nane"]]]["value"]


class block(Messanger):
    # https://github.com/pyro-ppl/numpyro/blob/master/numpyro/handlers.py#L206
    def __init__(self, fn: Callable, hide_fn: Callable = lambda msg: True) -> None:
        self.hide_fn = hide_fn
        super().__init__(fn=fn)

    def process_message(self, msg: Dict[str, Any]) -> None:
        if self.hide_fn(msg):
            msg["stop"] = True


class seed(Messanger):
    # https://github.com/pyro-ppl/numpyro/blob/master/numpyro/handlers.py#L611
    def __init__(self, fn: Callable, rng_seed: Union[int, jnp.ndarray]) -> None:
        self.rng_key = random.PRNGKey(rng_seed)
        super().__init__(fn=fn)

    def process_message(self, msg: Dict[str, Any]) -> None:
        if (
            (
                msg["type"] == "sample" and not msg["is_observed"]
                and msg["kwargs"]["rng_key"] is None
            )
            or msg["type"] in ["prng_key", "plate", "control_flow"]
        ):
            if msg["value"] is not None:
                return
            self.rng_key, rng_key_sample = random.split(self.rng_key)
            msg["kwargs"]["rng_key"] = rng_key_sample


class substitute(Messanger):
    # https://github.com/pyro-ppl/numpyro/blob/master/numpyro/handlers.py#L671
    def __init__(
        self, fn: Optional[Callable],
        data: Dict[str, np.ndarray],
        substitute_fn: Optional[Callable] = None,
    ) -> None:
        self.substitute_fn = substitute_fn
        self.data = data
        super().__init__(fn=fn)

    def process_message(self, msg: Dict[str, Any]) -> None:
        if (
            msg["type"] not in ("sample", "param", "plate")
            or msg.get("_control_flow_done", False)
        ):
            if msg["type"] == "control_flow":
                if self.data is not None:
                    msg["kwargs"]["substitute_stack"].append(("substitute", self.data))
                if self.substitute_fn is not None:
                    msg["kwargs"]["substitute_stack"].append(("substitute", self.substitute_fn))
            return

        if self.data is not None:
            value = self.data.get(msg["name"])
        else:
            value = self.substitute_fn(msg)
        
        if value is not None:
            msg["value"] = value


class plate(Messanger):
    # https://github.com/pyro-ppl/numpyro/blob/master/numpyro/primitives.py#L320
    def __init__(self, name: str, size: int, dim: Optional[int]) -> None:
        assert size > 0
        self.name = name
        self.size = size
        self.dim = dim
        super().__init__()

    def process_message(self, msg: Dict[str, Any]) -> None:
        if msg["type"] == "sample":
            batch_shape = msg["fn"].batch_shape
            if len(batch_shape) < -self.dim or batch_shape[self.dim] != self.size:
                batch_shape = [1] * (-self.dim - len(batch_shape)) + list(batch_shape)
                batch_shape[self.dim] = self.size
                msg["fn"] = msg["fn"].expand(batch_shape)

    def __iter__(self) -> Generator:
        return range(self.size)


def sample(
    name: str, fn: callable, obs: Optional[np.ndarray] = None, *args: Any, **kwargs: Any
) -> Any:
    # https://github.com/pyro-ppl/numpyro/blob/master/numpyro/primitives.py#L97

    if not NUMPYRO_STACK:
        return fn(*args, **kwargs)

    initial_msg = {
        "type": "sample",
        "name": name,
        "fn": fn,
        "args": args,
        "kwargs": kwargs,
        "value": obs,
    }

    msg = apply_stack(initial_msg)
    return msg["value"]


def prng_key() -> Any:
    # https://github.com/pyro-ppl/numpyro/blob/master/numpyro/primitives.py#L481

    if not NUMPYRO_STACK:
        return

    initial_msg = {
        "type": "prng_key",
        "fn": lambda rng_key: rng_key,
        "args": (),
        "kwargs": {"rng_key": None},
        "value": None,
    }

    msg = apply_stack(initial_msg)
    return msg["value"]


def identity(x: Any, *args: Any, **kwargs: Any) -> Any:
    # https://github.com/pyro-ppl/numpyro/blob/master/numpyro/util.py#L143
    return x


def param(
    name: str, init_value: Optional[Union[jnp.ndarray, Callable]] = None, **kwargs: Any
) -> Any:
    # https://github.com/pyro-ppl/numpyro/blob/master/numpyro/primitives.py#L152

    if not NUMPYRO_STACK:
        assert not callable(init_value)
        return init_value

    if callable(init_value):
        def fn(init_fn: Callable, *args, **kwargs):
            return init_fn(prng_key())

    initial_msg = {
        "type": "param",
        "name": name,
        "fn": fn,
        "args": (init_value,),
        "kwargs": kwargs,
        "value": None,
        "scale": None,
        "cond_indep_stack": [],
    }

    msg = apply_stack(initial_msg)
    return msg["value"]


def is_identically_one(x: Any) -> bool:
    # https://github.com/pyro-ppl/numpyro/blob/master/numpyro/distributions/util.py#L364
    if isinstance(x, (int, float)):
        return x == 1
    else:
        return False


def log_density(
    model: Callable, model_args: Tuple[Any, ...], model_kwargs: Dict[str, Any], params: jnp.ndarray
) -> Tuple[jnp.ndarray, OrderedDict]:
    # https://github.com/pyro-ppl/numpyro/blob/master/numpyro/infer/util.py#L36
    
    model = substitute(model, data=params)
    model_trace = trace(model).get_trace(*model_args, **model_kwargs)
    log_joint = jnp.zeros(())
    for site in model_trace.values():
        if site["type"] == "sample":
            value = site["value"]
            intermediates = site["intermediates"]
            scale = site["scale"]
            if intermediates:
                log_prob = site["fn"].log_prob(value, intermediates)
            else:
                log_prob = site["fn"].log_prob(value)

            if (scale is not None) and (not is_identically_one(scale)):
                log_prob = scale * log_prob

            log_prob = jnp.sum(log_prob)
            log_joint = log_joint + log_prob

    return log_joint, model_trace


class TraceELBO:
    # https://github.com/pyro-ppl/numpyro/blob/master/numpyro/infer/elbo.py#L17

    def __init__(self, num_particles: int = 1) -> None:
        self.num_particles = num_particles

    def loss(
        self,
        rng_key: np.ndarray,
        param_map: Dict[str, jnp.ndarray],
        model: Callable,
        guide: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> jnp.ndarray:
        def single_particle_elbo(rng_key: np.ndarray) -> jnp.ndarray:
            model_seed, guide_seed = random.split(rng_key)
            seeded_model = seed(model, model_seed)
            seeded_guide = seed(guide, guide_seed)
            guide_log_density, guide_trace = log_density(seeded_guide, args, kwargs, param_map)
            seeded_model = replay(seeded_model, guide_trace)
            model_log_density, _ = log_density(seeded_model, args, kwargs, param_map)

            elbo = model_log_density - guide_log_density
            return elbo

        if self.num_particles == 1:
            return -single_particle_elbo(rng_key)
        else:
            rng_keys = random.split(rng_key, self.num_particles)
            return -jnp.mean(vmap(single_particle_elbo)(rng_keys))
