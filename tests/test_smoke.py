"""Environment smoke tests: the numerical stack behaves as the project assumes."""

import jax
import jax.numpy as jnp
from jax import Array


def test_x64_is_enabled() -> None:
    assert jnp.zeros(1).dtype == jnp.float64


def test_prng_key_is_deterministic(key: Array) -> None:
    assert jnp.array_equal(jax.random.normal(key, (3,)), jax.random.normal(key, (3,)))
