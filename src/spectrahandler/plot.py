"""Figures for checking a resolution by eye. Needs the ``plot`` extra (matplotlib)."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from spectrahandler.curve_resolution.dataset import SpectralDataset
from spectrahandler.curve_resolution.diagnostics import NoiseDiagnostics
from spectrahandler.kinetics.fit import KineticFit

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = ["plot_kinetic_fit", "plot_noise"]

#: Diverging pair with a neutral midpoint: residuals below zero blue, above red.
_DIVERGING = ("#2a78d6", "#f0efec", "#e34948")
#: Categorical slots for series that are kinds, not signs: wavelength first, time second.
_CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a")
_INK, _MUTED, _SIGNAL, _NOISE = "#0b0b0b", "#52514e", "#2a78d6", "#a3a29b"
#: Species slots, fixed order, never cycled: a sixth species is an error, not a new hue.
_SPECIES = ("#2a78d6", "#eb6834", "#1baf7a", "#8c5bd6", "#c4950a")


def plot_noise(diagnostics: NoiseDiagnostics, dataset: SpectralDataset) -> "Figure":
    """Three panels: what the data hold, what is left, and whether what is left is noise.

    Left, the singular values against the white-noise edge: those above it carry
    signal. Middle, the residual after ``diagnostics.rank`` components over time and
    wavelength, scaled to +-3 sigma: white noise looks like even static, structure
    means something is missing. Right, the residual autocorrelation along wavelength
    and time: white noise drops to about zero after lag 0.

    Args:
        diagnostics: From :func:`spectrahandler.curve_resolution.noise_diagnostics`.
        dataset: The data the diagnostics were computed from, for the axes.

    Returns:
        A matplotlib figure with three axes.

    Raises:
        ImportError: If matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError as err:
        raise ImportError(
            "plot_noise needs matplotlib: install the 'plot' extra, "
            "e.g. uv add 'spectrahandler[plot]'"
        ) from err

    s = np.asarray(diagnostics.singular_values)
    shown = min(len(s), max(20, 3 * diagnostics.rank))
    fig, (ax_sv, ax_res, ax_ac) = plt.subplots(
        1, 3, figsize=(13, 3.8), width_ratios=(1, 1.5, 1), constrained_layout=True
    )

    index = np.arange(1, shown + 1)
    signal = s[:shown] > diagnostics.noise_edge
    ax_sv.semilogy(index, s[:shown], color=_MUTED, lw=1, zorder=1)
    ax_sv.scatter(
        index[signal],
        s[:shown][signal],
        s=28,
        color=_SIGNAL,
        zorder=2,
        label=f"signal ({int(signal.sum())})",
    )
    ax_sv.scatter(index[~signal], s[:shown][~signal], s=28, color=_NOISE, zorder=2, label="noise")
    ax_sv.axhline(diagnostics.noise_edge, color=_INK, lw=1, ls="--", label="white-noise edge")
    ax_sv.axvline(diagnostics.rank + 0.5, color=_MUTED, lw=0.8, ls=":")
    ax_sv.set(
        xlabel="component",
        ylabel="singular value (AU)",
        title=f"what the data hold\n{diagnostics.n_above_noise} components above noise; "
        f"rank {diagnostics.rank} kept",
    )
    ax_sv.legend(frameon=False, fontsize=8)

    residual = np.asarray(diagnostics.residual).reshape(-1, dataset.n_wavelength)
    limit = 3 * diagnostics.sigma
    cmap = LinearSegmentedColormap.from_list("residual", _DIVERGING)
    wl = np.asarray(dataset.wavelength)
    image = ax_res.imshow(
        residual,
        aspect="auto",
        cmap=cmap,
        vmin=-limit,
        vmax=limit,
        extent=(wl[0], wl[-1], residual.shape[0] - 0.5, -0.5),
        interpolation="nearest",
    )
    for boundary in range(1, dataset.n_run):
        ax_res.axhline(boundary * dataset.n_time - 0.5, color=_INK, lw=0.8)
    fig.colorbar(image, ax=ax_res, label="residual (AU)")
    ax_res.set(
        xlabel=f"wavelength ({dataset.wavelength_unit})",
        ylabel="timepoint (runs stacked)",
        title=f"what is left after {diagnostics.rank} components\n"
        f"sigma = {diagnostics.sigma:.2g} AU, colour scale ±3 sigma",
    )

    lags = np.arange(len(diagnostics.autocorrelation_wavelength))
    ax_ac.axhline(0, color=_MUTED, lw=0.8)
    ax_ac.plot(
        lags,
        np.asarray(diagnostics.autocorrelation_wavelength),
        marker="o",
        ms=4,
        color=_CATEGORICAL[0],
        lw=1.5,
        label="along wavelength",
    )
    ax_ac.plot(
        lags,
        np.asarray(diagnostics.autocorrelation_time),
        marker="s",
        ms=4,
        color=_CATEGORICAL[1],
        lw=1.5,
        label="along time",
    )
    ax_ac.set(
        xlabel="lag (channels or timepoints)",
        ylabel="autocorrelation",
        ylim=(-0.3, 1.05),
        title="is what is left noise?\nwhite noise: 1 at lag 0, then about 0",
    )
    ax_ac.legend(frameon=False, fontsize=8)

    for ax in (ax_sv, ax_res, ax_ac):
        title = ax.get_title()
        ax.set_title("")
        ax.set_title(title, loc="left", fontsize=9.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    return fig


def plot_kinetic_fit(
    fit: KineticFit, dataset: SpectralDataset, *, wavelengths: Sequence[float]
) -> "Figure":
    """Three panels: data against model, the species spectra, and what is left.

    Left, absorbance over time at up to three wavelengths, every run overlaid: dots are
    data, black lines the fitted model. Middle, each species' spectrum per run (thin)
    and their mean (thick). Right, the residual of every run divided by that run's
    fitted noise level per wavelength, runs stacked: even static means the model and
    the noise level explain the data; stripes or blocks mean something is missing.

    Args:
        fit: From :func:`spectrahandler.kinetics.fit_kinetics` on ``dataset``.
        dataset: The data that was fitted.
        wavelengths: One to three wavelengths for the left panel, in
            ``dataset.wavelength_unit``; the nearest channel is shown.

    Returns:
        A matplotlib figure with three axes.

    Raises:
        ValueError: If not one to three wavelengths are given, or there are more than
            five species.
        ImportError: If matplotlib is not installed.
    """
    if not 1 <= len(wavelengths) <= len(_CATEGORICAL):
        raise ValueError(f"give 1 to {len(_CATEGORICAL)} wavelengths, got {len(wavelengths)}")
    if len(fit.species) > len(_SPECIES):
        raise ValueError(f"plot_kinetic_fit draws at most {len(_SPECIES)} species")
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError as err:
        raise ImportError(
            "plot_kinetic_fit needs matplotlib: install the 'plot' extra, "
            "e.g. uv add 'spectrahandler[plot]'"
        ) from err

    fig, (ax_tr, ax_sp, ax_res) = plt.subplots(
        1, 3, figsize=(15, 4), width_ratios=(1.2, 1, 1.4), constrained_layout=True
    )
    wl = np.asarray(dataset.wavelength)
    mask = np.asarray(dataset.mask)
    time = np.asarray(dataset.time)
    absorbance = np.asarray(dataset.absorbance)
    residuals = np.asarray(fit.residuals)

    for j, target in enumerate(wavelengths):
        i = int(np.argmin(np.abs(wl - target)))
        for r in range(dataset.n_run):
            m = mask[r]
            ax_tr.plot(
                time[r, m],
                absorbance[r, m, i],
                "o",
                ms=2.5,
                mec="none",
                alpha=0.45,
                color=_CATEGORICAL[j],
                label=f"{wl[i]:.0f} {dataset.wavelength_unit}" if r == 0 else None,
            )
            ax_tr.plot(time[r, m], absorbance[r, m, i] - residuals[r, m, i], color=_INK, lw=1)
    ax_tr.set(
        xlabel=f"time ({dataset.time_unit})",
        ylabel="absorbance",
        title="data and model\ndots: data, lines: fitted model, runs overlaid",
    )
    ax_tr.legend(frameon=False, fontsize=8)

    spectra = np.asarray(fit.spectra)
    for s, name in enumerate(fit.species):
        for r in range(dataset.n_run):
            ax_sp.plot(wl, spectra[r, s], color=_SPECIES[s], lw=0.8, alpha=0.5)
        ax_sp.plot(wl, spectra[:, s].mean(axis=0), color=_SPECIES[s], lw=2, label=name)
    ax_sp.set(
        xlabel=f"wavelength ({dataset.wavelength_unit})",
        ylabel=f"absorbance per {dataset.concentration_unit}",
        title="species spectra\nthin: each run, thick: mean over runs",
    )
    ax_sp.legend(frameon=False, fontsize=8)

    scaled = residuals / np.asarray(fit.sigma)[:, None, :]
    stacked = np.concatenate([scaled[r, mask[r]] for r in range(dataset.n_run)])
    boundaries = np.cumsum(mask.sum(axis=1))[:-1]
    image = ax_res.imshow(
        stacked,
        aspect="auto",
        cmap=LinearSegmentedColormap.from_list("residual", _DIVERGING),
        vmin=-3,
        vmax=3,
        extent=(wl[0], wl[-1], stacked.shape[0] - 0.5, -0.5),
        interpolation="nearest",
    )
    for boundary in boundaries:
        ax_res.axhline(boundary - 0.5, color=_INK, lw=0.8)
    fig.colorbar(image, ax=ax_res, label="residual / noise level")
    ax_res.set(
        xlabel=f"wavelength ({dataset.wavelength_unit})",
        ylabel="timepoint (runs stacked)",
        title="what is left, per noise level\neven static: noise; stripes or blocks: misfit",
    )

    for ax in (ax_tr, ax_sp, ax_res):
        title = ax.get_title()
        ax.set_title("")
        ax.set_title(title, loc="left", fontsize=9.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    return fig
