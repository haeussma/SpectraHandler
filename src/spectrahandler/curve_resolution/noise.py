"""Noise that neighbouring wavelength channels share, measured from data.

Detectors and their software often average or interpolate along wavelength, so the noise of
neighbouring channels is not independent. Treating it as independent claims more
information than the data hold; binning hides the problem without measuring it. This module
measures the correlation from the data themselves, with no instrument parameter, so that a
fit can use every channel and still count the information correctly.
"""

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

__all__ = ["NoiseCorrelation", "measure_noise_correlation"]


@dataclass(frozen=True, eq=False)
class NoiseCorrelation:
    """How the noise of one spectrum is correlated along the channel index.

    The correlation is taken to depend only on the distance between two channels, not on the
    wavelength itself. It is held as a power spectrum, which is what a fit needs and what
    the data determine with the same relative precision at every frequency. It belongs to
    the channel spacing it was measured on.

    Attributes:
        frequency: Frequencies along the channel index, in cycles per channel from 0 to at
            most 0.5, shape ``(n_frequency,)``.
        power: Power of unit-variance noise at those frequencies, without the shared part,
            shape ``(n_frequency,)``. Independent channels give 1 everywhere; noise averaged
            over neighbouring channels has its power at low frequency.
        shared: Fraction of the noise variance that the whole spectrum has in common, such
            as a jitter of the lamp from one spectrum to the next.
        n_spectra: Number of noise spectra the measurement rests on.
    """

    frequency: Array
    power: Array
    shared: float
    n_spectra: int

    def spectrum(self, n_wavelength: int, floor: float = 1e-3) -> Array:
        """Power spectrum of unit-variance noise on a grid of ``n_wavelength`` channels.

        The grid is treated as circular (the Whittle approximation), which is accurate when
        the correlation reaches over far fewer channels than the grid has.

        Args:
            n_wavelength: Number of channels of the grid.
            floor: Smallest power allowed, relative to the mean, so that no frequency is
                trusted without limit.

        Returns:
            Shape ``(n_wavelength,)``, in the order of ``jnp.fft.fft``, with mean 1. The
            shared part sits at frequency zero.
        """
        k = jnp.arange(n_wavelength)
        frequency = jnp.minimum(k, n_wavelength - k) / n_wavelength
        power = jnp.interp(frequency, self.frequency, self.power)
        power = power.at[0].add(self.shared * n_wavelength)
        power = jnp.maximum(power, floor * power.mean())
        return power / power.mean()

    def correlation(self, n_lag: int) -> Array:
        """Correlation between two channels by their distance, for reading and plotting.

        Args:
            n_lag: Number of distances, starting at 0.

        Returns:
            Shape ``(n_lag,)``, first entry 1. Far apart, it levels off at ``shared``.
        """
        n = max(4 * n_lag, 2 * (self.frequency.shape[0] - 1))
        correlation: Array = jnp.fft.ifft(self.spectrum(n, floor=0.0)).real[:n_lag]
        return correlation


def measure_noise_correlation(absorbance: Array) -> NoiseCorrelation:
    """Measure how the noise of neighbouring wavelength channels is correlated.

    Noise is taken from the differences between consecutive spectra, which removes
    everything that does not change in time. So the sample must be static: a blank, or a
    stretch of a run in which the spectra no longer change, such as its end. A reaction in
    progress would be counted as noise that all channels share. Each channel is scaled by
    its own noise level, so only the correlation is measured, not the level, and the power
    spectra of all noise spectra are averaged.

    Args:
        absorbance: Shape ``(n_time, n_wavelength)`` or ``(n_run, n_time, n_wavelength)``,
            all finite, on equally spaced channels with the spacing of the data it will be
            used for. More spectra give a more precise measurement: the relative error of
            the power at each frequency is about ``1 / sqrt(n_spectra)``.

    Returns:
        The measured correlation.

    Raises:
        ValueError: If the input is not 2-D or 3-D or not finite, or if there are fewer
            than 16 channels or 3 spectra per run.
    """
    if absorbance.ndim not in (2, 3):
        raise ValueError(
            "absorbance must be (n_time, n_wavelength) or (n_run, n_time, n_wavelength), "
            f"got shape {absorbance.shape}"
        )
    if not bool(jnp.isfinite(absorbance).all()):
        raise ValueError("measure_noise_correlation requires finite absorbance")
    n_time, n_w = absorbance.shape[-2], absorbance.shape[-1]
    if n_w < 16 or n_time < 3:
        raise ValueError(f"need at least 16 channels and 3 spectra, got {n_w} and {n_time}")
    noise = jnp.diff(absorbance, axis=-2).reshape(-1, n_w)
    z = noise / noise.std(axis=0)
    power = (jnp.abs(jnp.fft.rfft(z, axis=1)) ** 2).mean(axis=0) / n_w
    # What the whole spectrum shares is a constant along wavelength: it sits at frequency
    # zero alone, on top of what the neighbouring frequencies show to be there anyway.
    nearby = power[1:3].mean()
    return NoiseCorrelation(
        frequency=jnp.fft.rfftfreq(n_w),
        power=power.at[0].set(nearby),
        shared=float(jnp.clip((power[0] - nearby) / n_w, 0.0, 1.0)),
        n_spectra=int(z.shape[0]),
    )
