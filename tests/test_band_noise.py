"""The band under noise that channels share, and under noise that grows with absorbance.

The correlated set is a realistic a -> b -> c run on 128 unbinned channels whose noise is
averaged over 8 neighbouring channels, with a reference for ``a``. The correlation is
measured on a blank with the same kind of noise.
"""

import dataclasses

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from spectrahandler.curve_resolution import (
    CredibleBand,
    SpectralDataset,
    make_correlated_noise,
    make_realistic_dataset,
    measure_noise_correlation,
    resolve_band,
)

SIGMA, N_W, WIDTH = 0.002, 128, 8


def _correlated_data(seed: int) -> tuple[SpectralDataset, Array, Array]:
    clean, spectra, concentrations = make_realistic_dataset(
        jax.random.key(0), noise=0.0, n_wavelength=N_W
    )
    k_data, k_ref = jax.random.split(jax.random.key(1000 + seed))
    noise = SIGMA * make_correlated_noise(k_data, clean.n_time, N_W, width=WIDTH)
    reference = spectra[0] + SIGMA / 50 * jax.random.normal(k_ref, (N_W,))
    data = dataclasses.replace(
        clean,
        absorbance=clean.absorbance + noise[None],
        reference_spectra=jnp.full((3, N_W), jnp.nan).at[0].set(reference),
        reference_sigma=jnp.array([SIGMA / 50, jnp.nan, jnp.nan]),
    )
    return data, spectra, concentrations


def _coverage(band: CredibleBand, spectra: Array, concentrations: Array) -> tuple[float, float]:
    """Fraction of true amounts and spectrum values inside the band; b and c matched by spectrum."""
    fitted = band.spectra_draws.mean(axis=0)[0]
    swapped = float(fitted[1] @ spectra[2]) > float(fitted[1] @ spectra[1])
    p = jnp.array([0, 2, 1]) if swapped else jnp.arange(3)
    c_in = (concentrations[..., p] >= band.concentration_lower) & (
        concentrations[..., p] <= band.concentration_upper
    )
    s_in = (spectra[p] >= band.spectra_lower[0]) & (spectra[p] <= band.spectra_upper[0])
    return float(c_in.mean()), float(s_in.mean())


def test_measured_correlation_keeps_the_truth_inside_where_independence_fails() -> None:
    """Assuming independent channels counts the same noise several times."""
    data, spectra, concentrations = _correlated_data(0)
    blank = SIGMA * make_correlated_noise(jax.random.key(99), 100, N_W, width=WIDTH)
    correlation = measure_noise_correlation(blank)
    band = resolve_band(data, jax.random.key(0), noise_correlation=correlation)
    amounts, values = _coverage(band, spectra, concentrations)
    assert amounts >= 0.95, f"band holds {amounts:.3f} of the true amounts"
    assert values >= 0.95, f"band holds {values:.3f} of the true spectrum values"
    assert band.r_hat < 1.05
    assert float(jnp.median(band.sigma)) == pytest.approx(SIGMA, rel=0.15)

    def width(result: CredibleBand) -> float:
        """Median width of the band of ``a``, the species the reference pins."""
        return float(
            jnp.median(result.concentration_upper[..., 0] - result.concentration_lower[..., 0])
        )

    # With the wrong noise model the true zeros can look significantly negative, so that no
    # split is feasible; if one is, its band is too narrow.
    try:
        independent = resolve_band(data, jax.random.key(0))
    except ValueError as error:
        assert "no feasible split" in str(error)
    else:
        assert width(band) > 1.1 * width(independent)


def test_growth_of_the_noise_with_absorbance_is_inferred() -> None:
    """Shot noise, exponent 0.5, on data that reach 1.3 AU: the posterior finds it."""
    clean, _, _ = make_realistic_dataset(jax.random.key(0), noise=0.0, total_concentration=50.0)
    a = clean.absorbance
    sd = SIGMA * 10.0 ** (0.5 * (a - a.mean(axis=1, keepdims=True)))
    data = dataclasses.replace(
        clean, absorbance=a + sd * jax.random.normal(jax.random.key(5), a.shape)
    )
    band = resolve_band(data, jax.random.key(1), noise_exponent=None, n_iter=300, n_burn=100)
    draws = band.noise_exponent_draws
    assert draws.shape == (1000,)
    assert float(draws.mean()) == pytest.approx(0.5, abs=0.2)
    assert float(draws.std()) < 0.15
    # sigma is the noise at each wavelength's mean absorbance.
    assert float(jnp.median(band.sigma)) == pytest.approx(SIGMA, rel=0.15)


def test_a_given_exponent_is_recorded() -> None:
    data, _, _ = make_realistic_dataset(jax.random.key(0))
    plain = resolve_band(data, jax.random.key(1), n_iter=200, n_burn=100)
    assert bool((plain.noise_exponent_draws == 0.0).all())
    shot = resolve_band(data, jax.random.key(1), noise_exponent=0.5, n_iter=200, n_burn=100)
    assert bool((shot.noise_exponent_draws == 0.5).all())
