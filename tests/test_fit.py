"""Per-run maximum likelihood: recovery, what is absorbed, ambiguity, rank, masks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spectrahandler.curve_resolution import SpectralDataset
from spectrahandler.kinetics import Scheme, fit_kinetics, make_kinetic_replicates

ONE_STEP = Scheme(steps=[("A", "B")])
TWO_STEP = Scheme(steps=[("A", "B"), ("B", "C")])


def test_recovers_rates_and_spectra_from_nearly_clean_data(key: jax.Array) -> None:
    data, log_rates, spectra = make_kinetic_replicates(
        key,
        TWO_STEP,
        {("A", "B"): 0.6, ("B", "C"): 0.25},
        initial={"A": 10.0},
        n_replicates=1,
        noise=1e-7,
    )
    fit = fit_kinetics(data, TWO_STEP, initial_rates={("A", "B"): 0.5, ("B", "C"): 0.2})
    assert fit.converged == (True,)
    np.testing.assert_allclose(fit.rates[0], jnp.exp(log_rates[0]), rtol=1e-5)
    np.testing.assert_allclose(fit.spectra[0], spectra, rtol=1e-4, atol=1e-7)


def test_shapes_and_units(key: jax.Array) -> None:
    data, _, _ = make_kinetic_replicates(key, ONE_STEP, {("A", "B"): 0.8}, initial={"A": 10.0})
    fit = fit_kinetics(data, ONE_STEP)
    assert fit.rates.shape == (4, 1)
    assert fit.spectra.shape == (4, 2, data.n_wavelength)
    assert fit.concentrations.shape == (4, data.n_time, 2)
    assert fit.residuals.shape == (4, data.n_time, data.n_wavelength)
    assert fit.sigma.shape == (4, data.n_wavelength)
    assert fit.log_likelihood.shape == (4,)
    assert fit.time_unit == "s"
    assert all(fit.converged)
    assert bool(jnp.isfinite(fit.log_rate_sd).all())


def _shifted(
    data: SpectralDataset, absorbance: jax.Array | None = None, time: jax.Array | None = None
) -> SpectralDataset:
    return SpectralDataset.create(
        absorbance=data.absorbance if absorbance is None else absorbance,
        time=data.time if time is None else time,
        wavelength=data.wavelength,
        species=data.species,
        initial_state=data.initial_state,
        run_ids=data.run_ids,
        conditions=data.conditions,
        time_unit=data.time_unit,
    )


def test_static_offset_and_time_shift_do_not_move_the_rates(key: jax.Array) -> None:
    """A static offset and a time shift are absorbed by the species spectra."""
    data, _, _ = make_kinetic_replicates(
        key, TWO_STEP, {("A", "B"): 0.6, ("B", "C"): 0.25}, initial={"A": 10.0}
    )
    rates = {("A", "B"): 0.5, ("B", "C"): 0.2}
    base_fit = fit_kinetics(data, TWO_STEP, initial_rates=rates)
    base = base_fit.rates
    tilt = 0.01 + 0.02 * jnp.linspace(-1.0, 1.0, data.n_wavelength)
    offset = fit_kinetics(
        _shifted(data, absorbance=data.absorbance + tilt), TWO_STEP, initial_rates=rates
    ).rates
    duration = float(data.time[0, -1] - data.time[0, 0])
    shifted = fit_kinetics(
        _shifted(data, time=data.time + 0.02 * duration), TWO_STEP, initial_rates=rates
    )
    np.testing.assert_allclose(offset, base, rtol=1e-6)
    np.testing.assert_allclose(shifted.rates, base, rtol=1e-6)
    # End product "C" is unchanged by the shift; "A" and "B" absorb it.
    np.testing.assert_allclose(shifted.spectra[:, 2], base_fit.spectra[:, 2], rtol=1e-6, atol=1e-9)


def test_rate_swap_is_reported_for_a_consecutive_scheme(key: jax.Array) -> None:
    data, _, _ = make_kinetic_replicates(
        key, TWO_STEP, {("A", "B"): 0.6, ("B", "C"): 0.25}, initial={"A": 10.0}, n_replicates=1
    )
    fit = fit_kinetics(data, TWO_STEP, initial_rates={("A", "B"): 0.5, ("B", "C"): 0.2})
    assert len(fit.ambiguities) == 1
    run_id, alternative = fit.ambiguities[0]
    assert run_id == "replicate1"
    np.testing.assert_allclose(alternative, np.asarray(fit.rates[0])[::-1], rtol=1e-12)


def test_one_step_has_no_ambiguity(key: jax.Array) -> None:
    data, _, _ = make_kinetic_replicates(key, ONE_STEP, {("A", "B"): 0.8}, initial={"A": 10.0})
    assert fit_kinetics(data, ONE_STEP).ambiguities == ()


def test_padded_run_gives_the_same_rates(key: jax.Array) -> None:
    data, _, _ = make_kinetic_replicates(
        key, ONE_STEP, {("A", "B"): 0.8}, initial={"A": 10.0}, n_replicates=1
    )
    padded = SpectralDataset.create(
        absorbance=jnp.concatenate(
            [data.absorbance, jnp.full((1, 5, data.n_wavelength), jnp.nan)], axis=1
        ),
        time=jnp.concatenate([data.time, jnp.full((1, 5), data.time[0, -1])], axis=1),
        mask=jnp.concatenate([jnp.ones((1, data.n_time), bool), jnp.zeros((1, 5), bool)], axis=1),
        wavelength=data.wavelength,
        species=data.species,
        initial_state=data.initial_state,
        run_ids=data.run_ids,
        time_unit=data.time_unit,
    )
    fit, fit_padded = fit_kinetics(data, ONE_STEP), fit_kinetics(padded, ONE_STEP)
    np.testing.assert_allclose(fit_padded.rates, fit.rates, rtol=1e-10)
    assert bool(jnp.isnan(fit_padded.residuals[0, -5:]).all())


def test_species_never_populated_is_rank_deficient(key: jax.Array) -> None:
    data, _, _ = make_kinetic_replicates(
        key, ONE_STEP, {("A", "B"): 0.8}, initial={"A": 10.0}, n_replicates=1
    )
    extra = Scheme(steps=[("A", "B"), ("C", "B")])
    with_c = SpectralDataset.create(
        absorbance=data.absorbance,
        time=data.time,
        wavelength=data.wavelength,
        species=("A", "B", "C"),
        initial_state=jnp.array([[10.0, 0.0, 0.0]]),
        run_ids=data.run_ids,
        time_unit=data.time_unit,
    )
    with pytest.raises(ValueError, match=r"rank-deficient.*\['C'\]"):
        fit_kinetics(with_c, extra, initial_rates={("A", "B"): 0.8, ("C", "B"): 0.3})


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"scheme": Scheme(steps=[("A", "C")])}, "differ"),
        ({"initial_rates": {("A", "C"): 1.0}}, "exactly the steps"),
        ({"initial_rates": {("A", "B"): -1.0}}, "positive"),
    ],
)
def test_rejects(key: jax.Array, kwargs: dict[str, object], message: str) -> None:
    data, _, _ = make_kinetic_replicates(
        key, ONE_STEP, {("A", "B"): 0.8}, initial={"A": 10.0}, n_replicates=1
    )
    arguments: dict[str, object] = {"scheme": ONE_STEP}
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        fit_kinetics(data, **arguments)  # ty: ignore[invalid-argument-type]


def test_requires_float64(key: jax.Array) -> None:
    data, _, _ = make_kinetic_replicates(
        key, ONE_STEP, {("A", "B"): 0.8}, initial={"A": 10.0}, n_replicates=1
    )
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(RuntimeError, match="jax_enable_x64"):
            fit_kinetics(data, ONE_STEP)
    finally:
        jax.config.update("jax_enable_x64", True)
