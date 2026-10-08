"""Invariants of resolve_band that hold on any data, independent of coverage."""

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spectrahandler.curve_resolution import (
    FeasibleBand,
    SpectralDataset,
    make_realistic_dataset,
    resolve_band,
)

#: The realistic set's noise, and the concentration (uM) of the simulated reference scan.
SIGMA, REF_CONC = 0.002, 50.0


def _with_reference(dataset: SpectralDataset, spectrum_a: jax.Array) -> SpectralDataset:
    """The same data plus a reference spectrum for species ``a`` only."""
    refs = jnp.full((dataset.n_species, dataset.n_wavelength), jnp.nan).at[0].set(spectrum_a)
    return SpectralDataset.create(
        absorbance=dataset.absorbance,
        time=dataset.time,
        wavelength=dataset.wavelength,
        species=dataset.species,
        initial_state=dataset.initial_state,
        run_ids=dataset.run_ids,
        reference_spectra=refs,
        reference_sigma=jnp.array([SIGMA / REF_CONC, jnp.nan, jnp.nan]),
    )


@pytest.fixture(scope="module")
def resolved() -> tuple[SpectralDataset, FeasibleBand]:
    data, spectra, _ = make_realistic_dataset(jax.random.key(0))
    data = _with_reference(data, spectra[0])
    return data, resolve_band(data, jax.random.key(1), n_iter=2000)


def test_shapes(resolved: tuple[SpectralDataset, FeasibleBand]) -> None:
    data, band = resolved
    c_shape = (data.n_run, data.n_time, data.n_species)
    s_shape = (1, data.n_species, data.n_wavelength)  # one set of spectra for all runs
    assert band.concentration_lower.shape == band.concentration_upper.shape == c_shape
    assert band.spectra_lower.shape == band.spectra_upper.shape == s_shape
    assert band.concentration_ambiguity.shape == (2, *c_shape)
    assert band.spectra_ambiguity.shape == (2, *s_shape)
    assert band.concentration_draws.shape[1:] == c_shape
    assert band.spectra_draws.shape[1:] == s_shape


def test_one_reference_leaves_four_free_numbers(
    resolved: tuple[SpectralDataset, FeasibleBand],
) -> None:
    """9 entries of T, minus 3 for closure, minus 3 for the reference, plus 1 shared."""
    assert resolved[1].n_free == 4


def test_no_reference_leaves_six_free_numbers() -> None:
    data, _, _ = make_realistic_dataset(jax.random.key(0))
    assert resolve_band(data, jax.random.key(1), n_iter=200, n_burn=100).n_free == 6


def test_sigma_is_estimated(resolved: tuple[SpectralDataset, FeasibleBand]) -> None:
    assert resolved[1].sigma == pytest.approx(SIGMA, rel=0.1)


def test_band_brackets_ambiguity_and_is_non_negative(
    resolved: tuple[SpectralDataset, FeasibleBand],
) -> None:
    _, band = resolved
    assert bool((band.concentration_lower >= 0).all())
    assert bool((band.spectra_lower >= 0).all())
    # The ambiguity extremes may sit a little below zero (the noise slack); the band is
    # clipped at zero, so compare against the clipped extreme.
    floor = jnp.clip(band.concentration_ambiguity[0], 0.0)
    assert bool((band.concentration_lower <= floor + 1e-12).all())
    assert bool((band.concentration_upper >= band.concentration_ambiguity[1]).all())
    assert bool((band.spectra_upper >= band.spectra_ambiguity[1]).all())


def test_draws_respect_closure(resolved: tuple[SpectralDataset, FeasibleBand]) -> None:
    """Closure holds in least squares over the rows, so single rows scatter at noise level.

    Measured: overall mean 12.497 uM, rows between 12.31 and 12.71 uM.
    """
    data, band = resolved
    totals = band.concentration_draws.sum(axis=-1).mean(axis=0)
    total = float(data.initial_state.sum())
    assert float(totals.mean()) == pytest.approx(total, abs=0.02)
    assert float(jnp.abs(totals - total).max()) < 0.3


def test_draws_reproduce_the_data(resolved: tuple[SpectralDataset, FeasibleBand]) -> None:
    """Every split reproduces the data to the noise: residual rms close to sigma."""
    data, band = resolved
    predicted = (band.concentration_draws @ band.spectra_draws).mean(axis=0)
    rms = float(jnp.sqrt(((predicted - data.absorbance) ** 2).mean()))
    assert rms < 1.5 * SIGMA


def test_reference_lies_inside_its_band(
    resolved: tuple[SpectralDataset, FeasibleBand],
) -> None:
    data, band = resolved
    ref = data.reference_spectra[0]
    assert bool(((ref >= band.spectra_lower[0, 0]) & (ref <= band.spectra_upper[0, 0])).all())


def test_requires_fully_measured_runs() -> None:
    data, _, _ = make_realistic_dataset(jax.random.key(0))
    masked = SpectralDataset.create(
        absorbance=data.absorbance,
        time=data.time,
        wavelength=data.wavelength,
        species=data.species,
        initial_state=data.initial_state,
        run_ids=data.run_ids,
        mask=data.mask.at[0, 3].set(False),
    )
    with pytest.raises(NotImplementedError, match="fully measured"):
        resolve_band(masked, jax.random.key(1))


def test_contradictory_reference_is_reported() -> None:
    """A negative reference cannot be a non-negative spectrum: no split is feasible."""
    data, spectra, _ = make_realistic_dataset(jax.random.key(0))
    data = _with_reference(data, -spectra[0])
    with pytest.raises(ValueError, match="no feasible split"):
        resolve_band(data, jax.random.key(1), n_restart=2, n_search=2000)


def test_result_does_not_depend_on_units() -> None:
    """Data in ~M and ~mAU give the same band as in uM and AU, up to the factors.

    The factors are powers of two near 1e-6 and 1e3, so the rescaled data are exact in
    floating point and the results agree to rounding. Decimal factors change the data by
    one ulp, and the chain, which branches on exact feasibility tests, turns that into a
    different but equally valid path. Measured with 1e-6 and 1e3: bands differ by up to
    9% of their largest value, against up to 4% for a change of key; sigma agrees
    exactly. That case must still resolve -- before normalisation it raised "no
    feasible split".
    """
    data, spectra, _ = make_realistic_dataset(jax.random.key(0))
    data = _with_reference(data, spectra[0])
    kwargs = {"n_iter": 300, "n_burn": 100}
    ref = resolve_band(data, jax.random.key(1), **kwargs)

    def rescaled(c: float, a: float) -> FeasibleBand:
        """Concentrations times ``c``, absorbance times ``a``, so spectra times ``a / c``."""
        converted = dataclasses.replace(
            data,
            absorbance=data.absorbance * a,
            initial_state=data.initial_state * c,
            reference_spectra=data.reference_spectra * (a / c),
            reference_sigma=data.reference_sigma * (a / c),
        )
        return resolve_band(converted, jax.random.key(1), **kwargs)

    c, a = 2.0**-20, 2.0**10
    band = rescaled(c, a)
    np.testing.assert_allclose(band.concentration_lower, ref.concentration_lower * c, rtol=1e-6)
    np.testing.assert_allclose(band.concentration_upper, ref.concentration_upper * c, rtol=1e-6)
    np.testing.assert_allclose(band.spectra_lower, ref.spectra_lower * (a / c), rtol=1e-6)
    np.testing.assert_allclose(band.spectra_upper, ref.spectra_upper * (a / c), rtol=1e-6)
    assert band.sigma == pytest.approx(ref.sigma * a, rel=1e-6)
    assert rescaled(1e-6, 1e3).sigma == pytest.approx(ref.sigma * 1e3, rel=1e-6)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sigma": 0.0}, "sigma must be finite and positive, got 0.0"),
        ({"sigma": -0.002}, "sigma must be finite and positive, got -0.002"),
        ({"sigma": float("nan")}, "sigma must be finite and positive, got nan"),
        ({"z_slack": -1.0}, r"z_slack must be >= 0, got -1.0"),
        ({"n_iter": 500, "n_burn": 500}, r"n_burn must be >= 0 and < n_iter = 500, got 500"),
        ({"n_burn": -1}, r"n_burn must be >= 0 and < n_iter = 8000, got -1"),
        ({"thin": 0}, r"thin must be >= 1, got 0"),
        ({"n_restart": 0}, r"n_restart must be >= 1, got 0"),
        ({"n_als": -1}, r"n_als must be >= 0, got -1"),
        ({"n_search": 0}, r"n_search must be >= 1, got 0"),
    ],
)
def test_rejects_bad_arguments(
    resolved: tuple[SpectralDataset, FeasibleBand], kwargs: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        resolve_band(resolved[0], jax.random.key(1), **kwargs)


def test_requires_float64(resolved: tuple[SpectralDataset, FeasibleBand]) -> None:
    """Without x64 the search fails as a bogus "no feasible split"; refuse up front."""
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(RuntimeError, match="jax_enable_x64"):
            resolve_band(resolved[0], jax.random.key(1))
    finally:
        jax.config.update("jax_enable_x64", True)
