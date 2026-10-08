"""Is the chosen number of species enough? Rank and noise diagnostics.

Every split the band reports reproduces the data through the same best rank-K fit, so
the fit cannot tell splits apart. What it can tell is whether K is right: if the residual
after K components is white noise at one level, K components are all the data hold. A
structured residual means something is missing -- a species, a baseline, a per-run
offset -- or that the noise is not what the band assumes.
"""

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from spectrahandler.curve_resolution.dataset import SpectralDataset

__all__ = ["NoiseDiagnostics", "noise_diagnostics"]


@dataclass(frozen=True)
class NoiseDiagnostics:
    """What is left after keeping ``rank`` components, and whether it looks like noise.

    Attributes:
        singular_values: Of the stacked time course, shape ``(n_singular,)``, in
            absorbance units, largest first. ``n_singular = min(n_run * n_time,
            n_wavelength)``.
        sigma_by_rank: Noise standard deviation implied by keeping ``r`` components,
            for ``r = 0 .. n_singular - 1``, in absorbance units.
        rank: Components kept for ``residual``.
        sigma: ``sigma_by_rank[rank]``.
        noise_edge: The largest singular value white noise at ``sigma`` would produce
            in a matrix of this shape, ``sigma * (sqrt(n_row) + sqrt(n_wavelength))``.
            Singular values above it carry signal.
        n_above_noise: Number of singular values above ``noise_edge``.
        residual: Data minus the rank-``rank`` fit, shape
            ``(n_run, n_time, n_wavelength)``, in absorbance units.
        autocorrelation_wavelength: Residual autocorrelation along wavelength for lags
            ``0 .. max_lag``, shape ``(max_lag + 1,)``. White noise: 1, then ~0.
        autocorrelation_time: Same along time, within each run.
    """

    singular_values: Array
    sigma_by_rank: Array
    rank: int
    sigma: float
    noise_edge: float
    n_above_noise: int
    residual: Array
    autocorrelation_wavelength: Array
    autocorrelation_time: Array


def _autocorrelation(x: Array, axis: int, max_lag: int) -> Array:
    """Autocorrelation of ``x`` along ``axis`` for lags ``0 .. max_lag``, pooled."""
    x = jnp.moveaxis(x, axis, -1)
    energy = (x**2).sum()
    # The Python loop runs over lag values (max_lag is small), not over data.
    return jnp.stack(
        [energy / energy]
        + [(x[..., :-lag] * x[..., lag:]).sum() / energy for lag in range(1, max_lag + 1)]
    )


def noise_diagnostics(
    dataset: SpectralDataset, rank: int | None = None, *, max_lag: int = 10
) -> NoiseDiagnostics:
    """Singular values, residual and residual autocorrelation after ``rank`` components.

    Args:
        dataset: Validated observations; ``mask`` must be all True.
        rank: Components to keep. ``None`` keeps ``dataset.n_species``.
        max_lag: Largest lag of the residual autocorrelations, in channels and in
            timepoints.

    Returns:
        The diagnostics; :func:`spectrahandler.plot.plot_noise` draws them.

    Raises:
        NotImplementedError: If ``dataset.mask`` has any False entry.
        ValueError: If ``rank`` leaves no degrees of freedom for the noise.
    """
    if not bool(dataset.mask.all()):
        raise NotImplementedError("noise_diagnostics requires fully measured runs")
    rank = dataset.n_species if rank is None else rank
    n_row = dataset.n_run * dataset.n_time
    n_singular = min(n_row, dataset.n_wavelength)
    if not 0 <= rank < n_singular:
        raise ValueError(f"rank must be in [0, {n_singular}), got {rank}")

    data = dataset.absorbance.reshape(n_row, dataset.n_wavelength)
    u, s, vt = jnp.linalg.svd(data, full_matrices=False)
    kept = jnp.arange(n_singular)
    discarded = jnp.cumsum((s**2)[::-1])[::-1]
    dof = (n_row - kept) * (dataset.n_wavelength - kept)
    sigma_by_rank = jnp.sqrt(discarded / dof)
    sigma = float(sigma_by_rank[rank])
    edge = sigma * (n_row**0.5 + dataset.n_wavelength**0.5)

    residual = (data - (u[:, :rank] * s[:rank]) @ vt[:rank]).reshape(dataset.absorbance.shape)
    return NoiseDiagnostics(
        singular_values=s,
        sigma_by_rank=sigma_by_rank,
        rank=rank,
        sigma=sigma,
        noise_edge=edge,
        n_above_noise=int((s > edge).sum()),
        residual=residual,
        autocorrelation_wavelength=_autocorrelation(residual, 2, max_lag),
        autocorrelation_time=_autocorrelation(residual, 1, max_lag),
    )
