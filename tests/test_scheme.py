"""Scheme validation and its closed-form concentration profiles."""

import jax.numpy as jnp
import numpy as np
import pytest

from spectrahandler.kinetics import Scheme
from spectrahandler.kinetics.scheme import concentrations, rate_matrix

TIME = jnp.linspace(0.0, 10.0, 50)


def test_species_are_sorted_and_steps_kept_in_order() -> None:
    scheme = Scheme(steps=[("B", "C"), ("A", "B")])
    assert scheme.species == ("A", "B", "C")
    assert scheme.steps == (("B", "C"), ("A", "B"))
    assert scheme.n_steps == 2


@pytest.mark.parametrize(
    ("steps", "message"),
    [
        ([], "at least one step"),
        ([("A", "A")], "change the species"),
        ([("A", "B"), ("A", "B")], "unique"),
        ([("A", "B", "C")], "pair"),
        ([("A", "")], "pair"),
        (["AB"], "pair"),
        (("AB", "CD"), "pair"),
        ([5], "pair"),
    ],
)
def test_rejects(steps: list[tuple[str, ...]], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Scheme(steps=steps)  # ty: ignore[invalid-argument-type]


def test_index_steps_follow_the_given_species_order() -> None:
    scheme = Scheme(steps=[("A", "B")])
    assert scheme.index_steps(("B", "A")) == ((1, 0),)
    with pytest.raises(ValueError, match="differ"):
        scheme.index_steps(("A", "C"))


def test_rate_matrix_columns_sum_to_zero() -> None:
    k = rate_matrix(jnp.array([0.6, 0.4, 0.1]), ((0, 1), (1, 2), (1, 0)), 3)
    np.testing.assert_allclose(k.sum(axis=0), 0.0, atol=1e-15)


def test_consecutive_matches_the_analytic_solution() -> None:
    c = concentrations(TIME, jnp.array([0.6, 0.4]), jnp.array([1.0, 0.0, 0.0]), ((0, 1), (1, 2)))
    a = jnp.exp(-0.6 * TIME)
    b = 0.6 / (0.4 - 0.6) * (jnp.exp(-0.6 * TIME) - jnp.exp(-0.4 * TIME))
    np.testing.assert_allclose(c[:, 0], a, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(c[:, 1], b, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(c.sum(axis=1), 1.0, rtol=1e-12)


def test_equal_rates_are_exact() -> None:
    """Equal rates make K defective; expm still gives b = k t exp(-k t)."""
    c = concentrations(TIME, jnp.array([0.5, 0.5]), jnp.array([1.0, 0.0, 0.0]), ((0, 1), (1, 2)))
    np.testing.assert_allclose(c[:, 1], 0.5 * TIME * jnp.exp(-0.5 * TIME), rtol=1e-10, atol=1e-14)


def test_reversible_pair_reaches_equilibrium() -> None:
    c = concentrations(
        jnp.array([200.0]), jnp.array([0.3, 0.1]), jnp.array([1.0, 0.0]), ((0, 1), (1, 0))
    )
    np.testing.assert_allclose(c[0], [0.25, 0.75], rtol=1e-10)
