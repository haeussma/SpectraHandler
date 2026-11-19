"""measure_noise_correlation: the correlation that went in comes back out."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spectrahandler.curve_resolution import (
    make_correlated_noise,
    make_realistic_dataset,
    measure_noise_correlation,
)

N_W = 256


def test_generated_noise_has_unit_variance_and_the_stated_correlation(key: jax.Array) -> None:
    noise = make_correlated_noise(key, 4000, 64, width=8, shared=0.1, white=0.1)
    np.testing.assert_allclose(noise.var(axis=0), 1.0, atol=0.12)
    lag4 = float((noise[:, :-4] * noise[:, 4:]).mean())
    assert lag4 == pytest.approx(0.1 + 0.8 * (1 - 4 / 8), abs=0.03)


def test_independent_noise_has_no_correlation(key: jax.Array) -> None:
    measured = measure_noise_correlation(0.002 * jax.random.normal(key, (200, N_W)))
    correlation = measured.correlation(20)
    assert float(correlation[0]) == pytest.approx(1.0)
    assert float(jnp.abs(correlation[1:]).max()) < 0.03
    assert measured.shared == pytest.approx(0.0, abs=0.01)
    assert measured.n_spectra == 199
    np.testing.assert_allclose(measured.power, 1.0, atol=0.35)


def test_a_moving_average_is_measured_from_a_blank(key: jax.Array) -> None:
    """Width 12, 5% shared, 5% white: a straight fall to the shared level at lag 12."""
    levels = 0.002 * jnp.linspace(1.0, 3.0, N_W)  # the noise level must not matter
    blank = levels * make_correlated_noise(key, 300, N_W, width=12, shared=0.05, white=0.05)
    measured = measure_noise_correlation(blank)
    lag = jnp.arange(1, 20)
    expected = 0.05 + 0.9 * jnp.clip(1 - lag / 12, 0.0)
    np.testing.assert_allclose(measured.correlation(20)[1:], expected, atol=0.03)
    assert measured.shared == pytest.approx(0.05, abs=0.02)


def test_a_reaction_in_progress_distorts_the_measurement(key: jax.Array) -> None:
    """The sample must be static: a changing signal is not removed, and it shows."""
    clean, _, _ = make_realistic_dataset(jax.random.key(0), noise=0.0, n_time=100, n_wavelength=N_W)
    noise = 0.002 * make_correlated_noise(key, 100, N_W, width=12, shared=0.05, white=0.05)
    running = measure_noise_correlation(clean.absorbance + noise[None])
    static = measure_noise_correlation(noise)
    assert float(jnp.abs(running.correlation(40) - static.correlation(40)).max()) > 0.1


def test_spectrum_is_positive_with_mean_one(key: jax.Array) -> None:
    blank = make_correlated_noise(key, 300, N_W, width=12)
    spectrum = measure_noise_correlation(blank).spectrum(N_W)
    assert spectrum.shape == (N_W,)
    assert float(spectrum.mean()) == pytest.approx(1.0)
    assert float(spectrum.min()) > 0
    assert float(spectrum[0]) > 10  # the shared part and the average both sit at low frequency
    np.testing.assert_allclose(spectrum[1:], spectrum[1:][::-1])  # fft order: symmetric


def test_spectrum_adapts_to_another_grid_length(key: jax.Array) -> None:
    """The correlation is by channel distance, so it carries over to a longer or shorter grid."""
    measured = measure_noise_correlation(make_correlated_noise(key, 300, N_W, width=12))
    assert measured.spectrum(100).shape == (100,)
    assert float(measured.spectrum(1000).mean()) == pytest.approx(1.0)
    lag = jnp.arange(1, 12)
    for n in (100, 1000):
        correlation = jnp.fft.ifft(measured.spectrum(n)).real[1:12]
        np.testing.assert_allclose(correlation, 0.05 + 0.9 * (1 - lag / 12), atol=0.05)


@pytest.mark.parametrize(
    ("shape", "match"),
    [
        ((N_W,), "must be"),
        ((10, 8), "at least 16 channels and 3 spectra"),
        ((2, N_W), "at least 16 channels and 3 spectra"),
    ],
)
def test_rejects_bad_input(shape: tuple[int, ...], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        measure_noise_correlation(jnp.ones(shape))


def test_rejects_missing_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        measure_noise_correlation(jnp.full((10, N_W), jnp.nan))
