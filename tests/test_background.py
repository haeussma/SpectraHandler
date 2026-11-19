"""fit_backgrounds: amplitudes from the zero window, and the tolerance it leaves."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spectrahandler.curve_resolution import Background, fit_backgrounds

SIGMA = 0.002
WAVELENGTH = jnp.linspace(340.0, 700.0, 96)
WINDOW = ((660.0, 700.0),)
BOTH = (Background.offset(), Background.tilt())


def _signal(n_run: int = 2, n_time: int = 40) -> jax.Array:
    """A band at 450 nm that decays in time and is zero inside the window."""
    band = 0.3 * jnp.exp(-0.5 * ((WAVELENGTH - 450.0) / 30.0) ** 2)
    decay = jnp.exp(-jnp.linspace(0.0, 3.0, n_time))
    return jnp.tile(decay[:, None] * band, (n_run, 1, 1))


def test_shapes_of_the_provided_backgrounds() -> None:
    np.testing.assert_allclose(Background.offset().shape(WAVELENGTH), 1.0)
    tilt = Background.tilt().shape(WAVELENGTH)
    np.testing.assert_allclose([tilt[0], tilt[-1], tilt.mean()], [-0.5, 0.5, 0.0], atol=1e-12)


def test_offset_and_tilt_per_run_are_recovered(key: jax.Array) -> None:
    """Injected amplitudes come back within three standard errors, given a wide window.

    Over 600-700 nm (27 wavelengths, 40 spectra) the tilt has a standard error of about
    0.0008 AU and the offset, which is the line's value 130 nm outside the window, of
    about 0.0003 AU.
    """
    true = jnp.array([[0.005, 0.0108], [-0.005, -0.0108]])
    shapes = jnp.stack([b.shape(WAVELENGTH) for b in BOTH], axis=-1)
    clean = _signal()
    data = clean + (true @ shapes.T)[:, None, :] + SIGMA * jax.random.normal(key, clean.shape)
    fit = fit_backgrounds(data, WAVELENGTH, backgrounds=BOTH, zero_windows=((600.0, 700.0),))
    np.testing.assert_allclose(fit.amplitude[:, 0], true[:, 0], atol=3 * 0.0003)
    np.testing.assert_allclose(fit.amplitude[:, 1], true[:, 1], atol=3 * 0.0008)
    assert fit.corrected.shape == data.shape
    assert float(jnp.abs(fit.corrected - clean).mean()) < 2 * SIGMA


def test_tolerance_bounds_a_pattern_the_backgrounds_cannot_fit() -> None:
    """A fixed wiggle of 0.002 AU stays in the window, and the tolerance reports it."""
    wiggle = 0.002 * jnp.sin(2 * jnp.pi * WAVELENGTH / 15.0)
    fit = fit_backgrounds(
        _signal() + 0.004 + wiggle, WAVELENGTH, backgrounds=BOTH, zero_windows=WINDOW
    )
    assert 0.001 < fit.tolerance <= 0.002 + 1e-9
    run, where = fit.tolerance_at
    assert run in (0, 1)
    assert 660.0 <= where <= 700.0


def test_tolerance_factor_is_the_leverage_of_the_fit() -> None:
    """An offset carries a window error outwards unchanged; a tilt amplifies it with distance."""
    in_window = WAVELENGTH >= 660.0
    offset = fit_backgrounds(
        _signal(), WAVELENGTH, backgrounds=(Background.offset(),), zero_windows=WINDOW
    ).tolerance_factor
    np.testing.assert_allclose(offset[in_window], 1.0)
    np.testing.assert_allclose(offset[~in_window], 2.0)
    tilted = fit_backgrounds(
        _signal(), WAVELENGTH, backgrounds=BOTH, zero_windows=WINDOW
    ).tolerance_factor
    np.testing.assert_allclose(tilted[in_window], 1.0)
    assert bool((jnp.diff(tilted[~in_window]) < 0).all())  # grows away from the window
    assert float(tilted[0]) > 10.0


def test_clean_data_leave_no_tolerance() -> None:
    fit = fit_backgrounds(_signal() + 0.004, WAVELENGTH, backgrounds=BOTH, zero_windows=WINDOW)
    assert fit.tolerance == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(fit.amplitude[:, 0], 0.004, atol=1e-6)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"backgrounds": ()}, "at least one background and one zero window"),
        ({"zero_windows": ()}, "at least one background and one zero window"),
        ({"zero_windows": ((690.0, 720.0),)}, "is not inside the grid"),
        ({"zero_windows": ((690.0, 700.0),)}, "hold 3 wavelengths"),
        ({"smoothing": 0}, "smoothing must be >= 1"),
        (
            {"backgrounds": (Background.offset(), Background("again", jnp.ones_like))},
            "not separable",
        ),
        ({"backgrounds": (Background("short", lambda w: w[:5]),)}, "must be finite with shape"),
    ],
)
def test_rejects_bad_input(kwargs: dict[str, Any], match: str) -> None:
    arguments: dict[str, Any] = {"backgrounds": BOTH, "zero_windows": WINDOW} | kwargs
    with pytest.raises(ValueError, match=match):
        fit_backgrounds(_signal(), WAVELENGTH, **arguments)


def test_needs_a_run_axis() -> None:
    with pytest.raises(ValueError, match="n_run, n_time, n_wavelength"):
        fit_backgrounds(_signal()[0], WAVELENGTH, backgrounds=BOTH, zero_windows=WINDOW)
