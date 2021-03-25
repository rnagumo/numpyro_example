"""Mini-numpyro

This is re-implementation of mini-pyro with JAX. This library is independent of the rest of NumPyro
without `nunpyro.distributions` module.

http://pyro.ai/examples/minipyro.html
"""

import weakref
from collections import namedtuple, OrderedDict
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from jax import lax, random, tree_map, value_and_grad, vmap
from jax.experimental import optimizers
from numpyro.distributions import constraints


NUMPYRO_STACK = []


def apply_stack(msg: Dict[str, Any]) -> Dict[str, Any]:
    # https://github.com/pyro-ppl/numpyro/blob/master/numpyro/primitives.py#L21

    pointer = 0
    for pointer, handler in enumerate(reversed(NUMPYRO_STACK)):
        handler.process_message(msg)
        if msg.get("stop"):
            break

    if msg["value"] is None:
        if msg["type"] == "sample":
            msg["value"], msg["intermediates"] = msg["fn"](
                *msg["args"], sample_intermediates=True, **msg["kwargs"]
            )
        else:
            msg["value"] = msg["fn"](*msg["args"], **msg["kwargs"])

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
            msg["value"] = self.guide_trace[msg["name"]]["value"]


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
        if (
            isinstance(rng_seed, int)
            or (isinstance(rng_seed, jnp.ndarray) and not jnp.shape(rng_seed))
        ):
            rng_seed = random.PRNGKey(rng_seed)
        self.rng_key = rng_seed
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


CondIndepStackFrame = namedtuple("CondIndepStackFrame", ["name", "dim", "size"])


class plate(Messanger):
    # https://github.com/pyro-ppl/numpyro/blob/master/numpyro/primitives.py#L320
    def __init__(self, name: str, size: int, dim: Optional[int]) -> None:
        assert size > 0
        self.name = name
        self.size = size
        self.dim = dim
        super().__init__()

    @staticmethod
    def _get_batch_shape(cond_indep_stack: jnp.ndarray) -> Tuple[int, ...]:
        n_dims = max(-f.dim for f in cond_indep_stack)
        batch_shape = [1] * n_dims
        for f in cond_indep_stack:
            batch_shape[f.dim] = f.size
        return tuple(batch_shape)

    def process_message(self, msg: Dict[str, Any]) -> None:
        cond_indep_stack = msg["cond_indep_stack"]
        frame = CondIndepStackFrame(self.name, self.dim, self.size)
        cond_indep_stack.append(frame)
        if msg["type"] == "sample":
            expected_shape = self._get_batch_shape(cond_indep_stack)
            dist_batch_shape = msg["fn"].batch_shape
            if "sample_shape" in msg["kwargs"]:
                dist_batch_shape = msg["kwargs"]["sample_shape"] + dist_batch_shape
                msg["kwargs"]["sample_shape"] = ()

            overlap_idx = max(len(expected_shape) - len(dist_batch_shape), 0)
            trailing_shape = expected_shape[overlap_idx:]
            broadcast_shape = lax.broadcast_shapes(trailing_shape, tuple(dist_batch_shape))
            batch_shape = expected_shape[:overlap_idx] + broadcast_shape
            msg["fn"] = msg["fn"].expand(batch_shape)


def sample(
    name: str,
    fn: callable,
    obs: Optional[np.ndarray] = None,
    rng_key: Optional[jnp.ndarray] = None,
    sample_shape: Tuple[int, ...] = (),
    infer: Optional[Dict[str, Any]] = None,
    obs_mask: Optional[np.ndarray] = None,
) -> Any:
    # https://github.com/pyro-ppl/numpyro/blob/master/numpyro/primitives.py#L97

    if not NUMPYRO_STACK:
        return fn(rng_key=rng_key, sample_shape=sample_shape)

    if obs_mask is not None:
        raise NotImplementedError

    initial_msg = {
        "type": "sample",
        "name": name,
        "fn": fn,
        "args": (),
        "kwargs": {"rng_key": rng_key, "sample_shape": sample_shape},
        "value": obs,
        "scale": None,
        "is_observed": obs is not None,
        "intermediates": [],
        "cond_indep_stack": [],
        "infer": {} if infer is None else infer,
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
    else:
        fn = identity

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


class Trace_ELBO:
    # https://github.com/pyro-ppl/numpyro/blob/master/numpyro/infer/elbo.py#L17

    def __init__(self, num_particles: int = 1) -> None:
        self.num_particles = num_particles

    def loss(
        self,
        rng_key: jnp.ndarray,
        param_map: Dict[str, jnp.ndarray],
        model: Callable,
        guide: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> jnp.ndarray:
        def single_particle_elbo(rng_key: jnp.ndarray) -> jnp.ndarray:
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


class ConstraintRegistry:
    # https://github.com/pyro-ppl/numpyro/blob/master/numpyro/distributions/transforms.py#L839
    def __init__(self) -> None:
        self._registry = {}

    def register(self, constraint, factory: Optional[Callable] = None) -> Optional[Callable]:
        if factory is None:
            return lambda factory: self.register(constraint, factory)

        if isinstance(constraint, constraints.Constraint):
            constraint = type(constraint)

        self._registry[constraint] = factory

    def __call__(self, constraint) -> Any:
        try:
            factory = self._registry[type(constraint)]
        except KeyError as e:
            raise NotImplementedError from e

        return factory(constraint)


biject_to = ConstraintRegistry()


class Transform:
    # https://github.com/pyro-ppl/numpyro/blob/master/numpyro/distributions/transforms.py#L51
    domain = constraints.real
    codomain = constraints.real
    _inv = None

    @property
    def inv(self):
        inv = None
        if self._inv is not None:
            inv = self._inv()
        if inv is None:
            inv = _InverseTransform(self)
            self._inv = weakref.ref(inv)
        return inv

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def _inverse(self, y: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def log_abs_det_jacobian(
        self, x: jnp.ndarray, y: jnp.ndarray, intermediates: Any = None
    ) -> jnp.ndarray:
        raise NotImplementedError


class _InverseTransform(Transform):
    def __init__(self, transform: Transform) -> None:
        super().__init__()
        self._inv = transform

    @property
    def domain(self) -> Any:
        return self._inv.codomain

    @property
    def codomain(self) -> Any:
        return self._inv.domain

    @property
    def inv(self) -> Any:
        return self._inv

    def __call__(self, x: jnp.ndarray) -> Any:
        return self._inv._inverse(x)


class IdentityTransform(Transform):
    # https://github.com/pyro-ppl/numpyro/blob/master/numpyro/distributions/transforms.py#L450

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x

    def _inverse(self, y: jnp.ndarray) -> jnp.ndarray:
        return y

    def log_abs_det_jacobian(
        self, x: jnp.ndarray, y: jnp.ndarray, intermediates: Any = None
    ) -> jnp.ndarray:
        return jnp.zeros_like(x)


@biject_to.register(constraints.real)
def _transform_to_real(constraint) -> Any:
    # https://github.com/pyro-ppl/numpyro/blob/master/numpyro/distributions/transforms.py#L927
    return IdentityTransform()


_Params = TypeVar("_Params")
_OptState = TypeVar("_OptState")
_IterOptState = TypeVar("_IterOptState")


class _NumpyroOptim:
    # https://github.com/pyro-ppl/numpyro/blob/master/numpyro/optim.py#L37
    def __init__(self, optim_fn: Callable, *args: Any, **kwargs: Any) -> None:
        self.init_fn, self.update_fn, self.get_params_fn = optim_fn(*args, **kwargs)

    def init(self, params: _Params) -> _IterOptState:
        opt_state = self.init_fn(params)
        return jnp.array(0), opt_state

    def update(self, g: _Params, state: _IterOptState) -> _IterOptState:
        i, opt_state = state
        opt_state = self.update_fn(i, g, opt_state)
        return i + 1, opt_state

    def eval_and_update(self, fn: Callable, state: _IterOptState) -> _IterOptState:
        params = self.get_params(state)
        out, grads = value_and_grad(fn)(params)
        return out, self.update(grads, state)

    def get_params(self, state: _IterOptState) -> _Params:
        _, opt_state = state
        return self.get_params_fn(opt_state)


class Adam(_NumpyroOptim):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(optimizers.adam, *args, **kwargs)


SVIState = namedtuple("SVIState", ["optim_state", "rng_key"])


def _apply_loss_fn(
    loss_fn: Callable,
    rng_key: jnp.ndarray,
    constrain_fn: Callable,
    model: Callable,
    guide: Callable,
    args: Any,
    kwargs: Dict[str, Any],
    static_kwargs: Dict[str, Any],
    params: Dict[str, Any],
) -> Any:
    return loss_fn(rng_key, constrain_fn(params), model, guide, *args, **kwargs, **static_kwargs)


def transform_fn(
    transforms: Dict[str, Any], params: Dict[str, jnp.ndarray], invert: bool = False
) -> Dict[str, Any]:
    # https://github.com/pyro-ppl/numpyro/blob/master/numpyro/infer/util.py#L69

    if invert:
        transforms = {k: v.inv for k, v in transforms.items()}
    return {k: transforms[k](v) if k in transforms else v for k, v in params.items()}


class SVI:
    # https://github.com/pyro-ppl/numpyro/blob/master/numpyro/infer/svi.py#L37

    def __init__(
        self,
        model: Callable,
        guide: Callable,
        optim: _NumpyroOptim,
        loss: Trace_ELBO,
        **static_kwargs: Any,
    ) -> None:

        self.model = model
        self.guide = guide
        self.optim = optim
        self.loss = loss
        self.static_kwargs = static_kwargs
        self.constrain_fn = None

    def init(self, rng_key: jnp.ndarray, *args: Any, **kwargs: Any) -> SVIState:
        rng_key, model_seed, guide_seed = random.split(rng_key, 3)
        model_init = seed(self.model, model_seed)
        guide_init = seed(self.guide, guide_seed)
        guide_trace = trace(guide_init).get_trace(*args, **kwargs, **self.static_kwargs)
        model_trace = trace(replay(model_init, guide_trace)).get_trace(
            *args, **kwargs, **self.static_kwargs
        )

        params = {}
        inv_transforms = {}
        for site in list(model_trace.values()) + list(guide_trace.values()):
            if site["type"] == "param":
                constraint = site["kwargs"].pop("constraint", constraints.real)
                transform = biject_to(constraint)
                inv_transforms[site["name"]] = transform
                params[site["name"]] = transform.inv(site["value"])

        self.constrain_fn = partial(transform_fn, inv_transforms)
        params = tree_map(lambda x: lax.convert_element_type(x, jnp.result_type(x)), params)

        return SVIState(self.optim.init(params), rng_key)

    def update(
        self, svi_state: SVIState, *args: Any, **kwargs: Any
    ) -> Tuple[SVIState, jnp.ndarray]:

        rng_key, rng_key_step = random.split(svi_state.rng_key)
        loss_fn = partial(
            _apply_loss_fn, self.loss.loss, rng_key_step, self.constrain_fn, self.model,
            self.guide, args, kwargs, self.static_kwargs
        )
        loss_val, optim_state = self.optim.eval_and_update(loss_fn, svi_state.optim_state)

        return SVIState(optim_state, rng_key), loss_val


def main(num_epochs: int = 5) -> None:
    # https://github.com/pyro-ppl/pyro/blob/dev/examples/minipyro.py

    def model(data: jnp.ndarray) -> None:
        loc = sample("loc", dist.Normal(0, 1))
        with plate("data", len(data), dim=-1):
            sample("obs", dist.Normal(loc, 1), obs=data)

    def guide(data: jnp.ndarray) -> None:
        guide_loc = param("guide_loc", jnp.array(0.0))
        guide_scale = jnp.exp(param("guide_scale_log", jnp.array(0.0)))
        sample("loc", dist.Normal(guide_loc, guide_scale))

    rng_key = random.PRNGKey(0)
    rng_key, rng_key_init = random.split(rng_key)

    data = jnp.array(np.random.randn(100) + 3.0)
    svi = SVI(model, guide, Adam(0.001), Trace_ELBO())
    svi_state = svi.init(rng_key_init, data)

    for epoch in range(1, num_epochs + 1):
        svi_state, loss = svi.update(svi_state, data)
        print(epoch, loss)


if __name__ == "__main__":
    main()
