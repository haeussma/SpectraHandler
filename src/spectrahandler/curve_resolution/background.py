"""Backgrounds: known spectral shapes with fitted amplitudes, pinned by a zero window.

A baseline with a free spectrum cannot be told apart from the species spectra. A background
with a known shape can, if there is a wavelength range where no chemical species absorbs:
there the data hold only background and noise, so the amplitudes are measured there and the
known shapes carry them across the rest of the spectrum. What is left in the window after
the fit bounds the baseline error. See ADR 0004.
"""

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

__all__ = ["Background", "BackgroundFit", "align_baselines", "background_shapes", "fit_backgrounds"]


@dataclass(frozen=True)
class Background:
    """A known spectral shape whose amplitude is fitted, one amplitude per run.

    Attributes:
        name: Label under which the amplitude is reported.
        shape: Maps the wavelength grid, shape ``(n_wavelength,)``, to the unitless shape on
            that grid, same shape. The amplitude is in absorbance units.
    """

    name: str
    shape: Callable[[Array], Array]

    @classmethod
    def offset(cls) -> "Background":
        """A flat offset: the same absorbance at every wavelength."""
        return cls("offset", jnp.ones_like)

    @classmethod
    def tilt(cls) -> "Background":
        """A straight line through zero at the grid's centre.

        It runs from -0.5 at the first wavelength to +0.5 at the last, so its amplitude is
        the absorbance difference across the whole grid.
        """
        return cls("tilt", lambda w: (w - 0.5 * (w[0] + w[-1])) / (w[-1] - w[0]))


@dataclass(frozen=True, eq=False)
class BackgroundFit:
    """Backgrounds fitted in the zero windows, and what they left behind there.

    Attributes:
        corrected: The input minus the fitted backgrounds, shape
            ``(n_run, n_time, n_wavelength)``, in absorbance units.
        amplitude: Fitted amplitudes, shape ``(n_run, n_background)``, in absorbance units.
        tolerance: The baseline tolerance inside the zero windows: the largest
            time-smoothed leftover there over every run, time and window wavelength, in
            absorbance units. It bounds a static pattern and a drift together.
        tolerance_at: Where it was found: the run index and the wavelength.
        tolerance_factor: What ``tolerance`` must be multiplied by at each wavelength to
            bound the baseline error there, shape ``(n_wavelength,)``. It is 1 inside the
            windows. Outside it is ``1 + leverage``: a leftover of size ``tolerance`` in
            the windows can bias the amplitudes, and the shapes carry that bias outwards,
            the further the larger for a tilt.
    """

    corrected: Array
    amplitude: Array
    tolerance: float
    tolerance_at: tuple[int, float]
    tolerance_factor: Array


def background_shapes(backgrounds: tuple[Background, ...], wavelength: Array) -> Array:
    """The backgrounds' shapes on a wavelength grid, shape ``(n_wavelength, n_background)``.

    Args:
        backgrounds: The backgrounds; may be empty.
        wavelength: Grid, shape ``(n_wavelength,)``.

    Returns:
        One column per background.

    Raises:
        ValueError: If a shape does not have the grid's size or is not finite.
    """
    n_w = wavelength.shape[0]
    if not backgrounds:
        return jnp.zeros((n_w, 0))
    columns = [jnp.asarray(b.shape(wavelength)) for b in backgrounds]
    if any(c.shape != (n_w,) or not bool(jnp.isfinite(c).all()) for c in columns):
        raise ValueError(
            f"every background shape must be finite with shape ({n_w},); got "
            f"{[c.shape for c in columns]}"
        )
    return jnp.stack(columns, axis=-1)


def align_baselines(absorbance: Array, shapes: Array, n_species: int, n_sweep: int = 30) -> Array:
    """Background amplitudes of every run relative to the first, by alternating fits.

    All runs share one set of species spectra, so a difference between two runs that is
    constant in time and has a background's shape cannot be chemistry. Alternately the
    data, with the current backgrounds removed, are approximated at rank ``n_species``,
    and the backgrounds are refitted to what that approximation leaves, run by run. This is
    a starting point; the fit samples the amplitudes afterwards.

    Args:
        absorbance: Shape ``(n_run, n_time, n_wavelength)``.
        shapes: Background shapes, shape ``(n_wavelength, n_background)``.
        n_species: Rank of the chemistry.
        n_sweep: Number of alternations.

    Returns:
        Shape ``(n_run, n_background)``; the first row is zero.
    """
    n_run, n_w = absorbance.shape[0], absorbance.shape[2]
    solve = jnp.linalg.pinv(shapes)

    def sweep(_: int, amplitude: Array) -> Array:
        corrected = (absorbance - (amplitude @ shapes.T)[:, None, :]).reshape(-1, n_w)
        u, s, vt = jnp.linalg.svd(corrected, full_matrices=False)
        low_rank = ((u[:, :n_species] * s[:n_species]) @ vt[:n_species]).reshape(absorbance.shape)
        fitted: Array = (absorbance - low_rank).mean(axis=1) @ solve.T
        return fitted - fitted[0]

    start = jnp.zeros((n_run, shapes.shape[1]))
    aligned: Array = jax.lax.fori_loop(0, n_sweep, sweep, start)
    return aligned


def fit_backgrounds(
    absorbance: Array,
    wavelength: Array,
    *,
    backgrounds: tuple[Background, ...],
    zero_windows: tuple[tuple[float, float], ...],
    sigma: float | Array | None = None,
    smoothing: int = 20,
    shared: bool = False,
) -> BackgroundFit:
    """Fit background amplitudes where no species absorbs, and subtract the backgrounds.

    Per run, the time-mean spectrum inside the zero windows is fitted by weighted least
    squares with the background shapes; the fitted backgrounds are subtracted at every
    wavelength. The amplitudes' own uncertainty is not reported: it is far below the
    tolerance, which is the baseline error that matters.

    With ``shared``, one set of amplitudes is fitted to the mean over all runs and
    subtracted from every run: for runs whose baselines have already been aligned.

    Args:
        absorbance: Shape ``(n_run, n_time, n_wavelength)``, all finite, in absorbance
            units. Pass a single blank as ``blank[None]``.
        wavelength: Grid, shape ``(n_wavelength,)``, strictly increasing.
        backgrounds: The shapes to fit; at least one.
        zero_windows: Wavelength intervals ``(low, high)``, in the grid's unit, where no
            chemical species absorbs; at least one. This is an assumption: the returned
            tolerance tells how well it holds.
        sigma: Noise standard deviation, one number or one per wavelength, used as
            ``1 / sigma`` weights. ``None`` weights every wavelength equally.
        smoothing: Number of consecutive spectra averaged before the leftover is measured,
            so that noise does not set the tolerance. Clipped to ``n_time``.
        shared: Fit one set of amplitudes for all runs instead of one per run.

    Returns:
        The corrected data, the amplitudes, the tolerance and its factor per wavelength.

    Raises:
        ValueError: If the input is not 3-D or not finite, if there is no background or no
            window, if a window is not inside the grid, if the windows hold fewer than
            ``n_background + 2`` wavelengths, if a shape has the wrong size or is not
            finite, if the shapes are not separable inside the windows, or if ``smoothing``
            is below 1.
    """
    if absorbance.ndim != 3:
        raise ValueError(
            f"absorbance must be (n_run, n_time, n_wavelength), got shape {absorbance.shape}"
        )
    if not bool(jnp.isfinite(absorbance).all()):
        raise ValueError("fit_backgrounds requires finite absorbance: fully measured runs")
    if not backgrounds or not zero_windows:
        raise ValueError("fit_backgrounds needs at least one background and one zero window")
    if smoothing < 1:
        raise ValueError(f"smoothing must be >= 1, got {smoothing}")
    n_time, n_w = absorbance.shape[1], absorbance.shape[2]
    in_window = jnp.zeros(n_w, dtype=bool)
    for low, high in zero_windows:
        if not float(wavelength[0]) <= low < high <= float(wavelength[-1]):
            raise ValueError(
                f"zero window ({low}, {high}) is not inside the grid "
                f"{float(wavelength[0]):g} to {float(wavelength[-1]):g}"
            )
        in_window = in_window | ((wavelength >= low) & (wavelength <= high))
    names = [b.name for b in backgrounds]
    if int(in_window.sum()) < len(backgrounds) + 2:
        raise ValueError(
            f"the zero windows hold {int(in_window.sum())} wavelengths; {len(backgrounds)} "
            f"backgrounds need at least {len(backgrounds) + 2}"
        )
    shapes = background_shapes(backgrounds, wavelength)
    weight = jnp.ones(n_w) if sigma is None else jnp.broadcast_to(1.0 / jnp.asarray(sigma), (n_w,))
    design = (shapes * weight[:, None])[in_window]
    if float(jnp.linalg.cond(design)) > 1e6:
        raise ValueError(f"backgrounds {names} are not separable in the zero windows; widen them")

    solve = jnp.linalg.pinv(design)  # weighted window data -> amplitudes
    mean_spectrum = absorbance.mean(axis=1)
    if shared:
        mean_spectrum = jnp.broadcast_to(mean_spectrum.mean(axis=0), mean_spectrum.shape)
    amplitude: Array = (mean_spectrum * weight)[:, in_window] @ solve.T
    corrected = absorbance - (amplitude @ shapes.T)[:, None, :]
    # A leftover bounded by 1 in the windows moves the fitted background at a wavelength by
    # at most the absolute row sum of the map from window data to background there.
    leverage = jnp.abs((shapes @ solve) * weight[in_window]).sum(axis=1)

    # Running mean over k spectra, by differences of the cumulative sum along time.
    k = min(smoothing, n_time)
    leftover = corrected[:, :, in_window]
    total = jnp.cumsum(jnp.concatenate([jnp.zeros_like(leftover[:, :1]), leftover], axis=1), axis=1)
    worst = (jnp.abs(total[:, k:] - total[:, :-k]) / k).max(axis=1)
    run, column = jnp.unravel_index(jnp.argmax(worst), worst.shape)
    return BackgroundFit(
        corrected=corrected,
        amplitude=amplitude,
        tolerance=float(worst.max()),
        tolerance_at=(int(run), float(wavelength[in_window][column])),
        tolerance_factor=jnp.where(in_window, 1.0, 1.0 + leverage),
    )
