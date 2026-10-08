"""Figures for checking data and results by eye, drawn with matplotlib.

Every ``plot_*`` function except :func:`plot_noise` draws one run, picked by its id, on axes
you pass or on new ones, and returns those axes: place the panels in your own grid, one run
per axes. No function sets a title; what a panel shows is in its axis labels, legend and the
run id written inside the axes. Fonts and sizes come from matplotlib's rcParams, so
``plt.style.context(...)`` restyles every panel.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.scipy.stats import norm
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.lines import Line2D

from spectrahandler.curve_resolution.band import FeasibleBand
from spectrahandler.curve_resolution.dataset import SpectralDataset
from spectrahandler.curve_resolution.diagnostics import NoiseDiagnostics
from spectrahandler.kinetics.fit import KineticFit
from spectrahandler.kinetics.scheme import Step

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = [
    "DIVERGING",
    "INK",
    "MUTED",
    "SPECIES_COLORS",
    "plot_deconvolution",
    "plot_noise",
    "plot_rate_posterior",
    "plot_residuals",
    "plot_spectra",
]

#: Species colours in fixed order, never cycled: a sixth species needs ``colors=``.
SPECIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#8c5bd6", "#c4950a")
#: Diverging pair with a neutral midpoint: below zero blue, above red.
DIVERGING = ("#2a78d6", "#f0efec", "#e34948")
#: Text and model lines.
INK = "#0b0b0b"
#: Reference lines and secondary marks.
MUTED = "#52514e"
#: Categorical slots for series that are kinds, not signs: wavelength first, time second.
_CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a")
_SIGNAL, _NOISE = "#2a78d6", "#a3a29b"


def _new_axes(n: int) -> list[Axes]:
    """``n`` axes side by side on a new figure.

    Args:
        n: Number of axes.

    Returns:
        The axes, left to right.
    """
    fig = plt.figure(figsize=(5.0 * n, 3.8), layout="constrained")
    return list(fig.subplots(1, n, squeeze=False)[0])


def _tag(ax: Axes, text: str) -> None:
    """Write a run id inside the axes, top left: a label, not a title.

    Args:
        ax: The axes.
        text: What to write.
    """
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1},
    )


def _species_style(
    species: Sequence[str],
    colors: Mapping[str, str] | None,
    labels: Mapping[str, str] | None,
) -> tuple[list[str], list[str]]:
    """Colour and legend label per species, in ``species`` order.

    Args:
        species: Species names, in dataset order.
        colors: Colour per species name; ``None`` takes :data:`SPECIES_COLORS` in order.
        labels: Legend label per species name; species left out keep their name.

    Returns:
        Colours and labels, one per species.

    Raises:
        ValueError: If there are more species than default colours and no ``colors``,
            or ``colors`` misses a species.
    """
    if colors is None:
        if len(species) > len(SPECIES_COLORS):
            raise ValueError(
                f"{len(species)} species but {len(SPECIES_COLORS)} default colours: pass colors="
            )
        colors = dict(zip(species, SPECIES_COLORS, strict=False))
    missing = [name for name in species if name not in colors]
    if missing:
        raise ValueError(f"colors has no entry for {missing}")
    labels = labels or {}
    return [colors[name] for name in species], [labels.get(name, name) for name in species]


def plot_noise(diagnostics: NoiseDiagnostics, dataset: SpectralDataset) -> Figure:
    """Three panels over all runs: what the data hold, what is left, whether it is noise.

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
    """
    s = np.asarray(diagnostics.singular_values)
    shown = min(len(s), max(20, 3 * diagnostics.rank))
    fig, (ax_sv, ax_res, ax_ac) = plt.subplots(
        1, 3, figsize=(13, 3.8), width_ratios=(1, 1.5, 1), constrained_layout=True
    )

    index = np.arange(1, shown + 1)
    signal = s[:shown] > diagnostics.noise_edge
    ax_sv.semilogy(index, s[:shown], color=MUTED, lw=1, zorder=1)
    ax_sv.scatter(
        index[signal],
        s[:shown][signal],
        s=28,
        color=_SIGNAL,
        zorder=2,
        label=f"signal ({int(signal.sum())})",
    )
    ax_sv.scatter(index[~signal], s[:shown][~signal], s=28, color=_NOISE, zorder=2, label="noise")
    ax_sv.axhline(diagnostics.noise_edge, color=INK, lw=1, ls="--", label="white-noise edge")
    ax_sv.axvline(
        diagnostics.rank + 0.5, color=MUTED, lw=0.8, ls=":", label=f"rank {diagnostics.rank} kept"
    )
    ax_sv.set(xlabel="component", ylabel="singular value (AU)")
    ax_sv.legend(frameon=False, fontsize=8)

    residual = np.asarray(diagnostics.residual).reshape(-1, dataset.n_wavelength)
    limit = 3 * diagnostics.sigma
    cmap = LinearSegmentedColormap.from_list("residual", DIVERGING)
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
        ax_res.axhline(boundary * dataset.n_time - 0.5, color=INK, lw=0.8)
    fig.colorbar(image, ax=ax_res, label=f"residual after {diagnostics.rank} components (AU)")
    ax_res.set(xlabel=f"wavelength ({dataset.wavelength_unit})", ylabel="timepoint (runs stacked)")

    lags = np.arange(len(diagnostics.autocorrelation_wavelength))
    ax_ac.axhline(0, color=MUTED, lw=0.8)
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
    )
    ax_ac.legend(frameon=False, fontsize=8)

    for ax in (ax_sv, ax_res, ax_ac):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    return fig


def _check_runs(result: KineticFit | FeasibleBand, dataset: SpectralDataset) -> None:
    """Raise if ``result`` was not made from the runs of ``dataset``, in the same order."""
    if isinstance(result, FeasibleBand):
        if result.concentration_lower.shape[0] != dataset.n_run:
            raise ValueError(
                f"the band has {result.concentration_lower.shape[0]} runs, "
                f"the dataset {dataset.n_run}"
            )
    elif result.run_ids != dataset.run_ids:
        raise ValueError(
            f"the fit's run_ids {list(result.run_ids)} differ from the dataset's "
            f"run_ids {list(dataset.run_ids)}"
        )


def plot_spectra(
    dataset: SpectralDataset, run: str, *, ax: Axes | None = None, log_time: bool = False
) -> Axes:
    """Every measured spectrum of one run, coloured by time, with a time colorbar.

    Args:
        dataset: Any dataset; nothing is fitted.
        run: The run's id, one of ``dataset.run_ids``.
        ax: Axes to draw on; ``None`` makes a new figure.
        log_time: Colour by time on a log scale. Spectra at ``t <= 0`` take the colour
            of the earliest positive time.

    Returns:
        The axes drawn on.

    Raises:
        ValueError: If ``run`` is not a run of ``dataset``, or ``log_time`` is set and
            the run has no positive time.
    """
    r = dataset.run_index(run)
    ax = ax if ax is not None else _new_axes(1)[0]

    measured = np.asarray(dataset.mask[r])
    time = np.asarray(dataset.time[r])[measured]
    absorbance = np.asarray(dataset.absorbance[r])[measured]
    wavelength = np.asarray(dataset.wavelength)
    if log_time:
        if not (time > 0).any():
            raise ValueError(f"run {run!r} has no positive time for a log colour scale")
        time = np.maximum(time, time[time > 0].min())
        scale: Normalize = LogNorm(vmin=time.min(), vmax=time.max())
    else:
        scale = Normalize(vmin=time.min(), vmax=time.max())
    lines = LineCollection(
        list(np.stack([np.broadcast_to(wavelength, absorbance.shape), absorbance], axis=-1)),
        cmap=LinearSegmentedColormap.from_list("time", ("#c9dcf3", "#0d3a73")),
        norm=scale,
        linewidths=0.7,
    )
    lines.set_array(time)
    ax.add_collection(lines)
    ax.autoscale_view()
    ax.figure.colorbar(lines, ax=ax, label=f"time ({dataset.time_unit})")
    ax.set(xlabel=f"wavelength ({dataset.wavelength_unit})", ylabel="absorbance (AU)")
    _tag(ax, run)
    return ax


def plot_deconvolution(
    result: KineticFit | FeasibleBand,
    dataset: SpectralDataset,
    run: str,
    *,
    axes: Sequence[Axes] | None = None,
    colors: Mapping[str, str] | None = None,
    labels: Mapping[str, str] | None = None,
    log_time: bool = False,
) -> tuple[Axes, Axes]:
    """One run split into species: concentrations over time and species spectra.

    For a :class:`~spectrahandler.kinetics.KineticFit`, dots are every measured spectrum
    projected onto the run's fitted spectra (:meth:`SpectralDataset.project`), what the
    spectra alone say; ink lines are the kinetic model. For a
    :class:`~spectrahandler.curve_resolution.FeasibleBand`, both panels shade the band:
    every split the data allow, no single curve.

    Args:
        result: The fit or the band of ``dataset``.
        dataset: The data that were resolved.
        run: The run's id, one of ``dataset.run_ids``.
        axes: Two axes, concentrations then spectra; ``None`` makes a new figure.
        colors: Colour per species name; ``None`` takes :data:`SPECIES_COLORS` in order.
        labels: Legend label per species name, e.g. with mathtext subscripts.
        log_time: Log time axis; points at ``t <= 0`` are then left out.

    Returns:
        The concentration axes and the spectra axes.

    Raises:
        ValueError: If ``run`` is not a run of ``dataset``, if ``result`` was not made from
            the runs of ``dataset`` (a fit's ``run_ids`` differ, or a band has another
            number of runs), or for the colour errors of ``colors``.
    """
    r = dataset.run_index(run)
    _check_runs(result, dataset)
    ax_c, ax_s = axes if axes is not None else _new_axes(2)
    color, label = _species_style(dataset.species, colors, labels)
    measured = np.asarray(dataset.mask[r])
    if log_time:
        measured = measured & (np.asarray(dataset.time[r]) > 0)
    time = np.asarray(dataset.time[r])[measured]
    wavelength = np.asarray(dataset.wavelength)

    if isinstance(result, FeasibleBand):
        c_lo = np.asarray(result.concentration_lower[r])[measured]
        c_hi = np.asarray(result.concentration_upper[r])[measured]
        s_lo, s_hi = np.asarray(result.spectra_lower[0]), np.asarray(result.spectra_upper[0])
        for s in range(dataset.n_species):
            ax_c.fill_between(time, c_lo[:, s], c_hi[:, s], color=color[s], alpha=0.35, lw=0)
            ax_s.fill_between(
                wavelength, s_lo[s], s_hi[s], color=color[s], alpha=0.35, lw=0, label=label[s]
            )
        ax_c.legend(*ax_s.get_legend_handles_labels(), frameon=False)
    else:
        projected = np.asarray(dataset.project(result.spectra)[r])[measured]
        model = np.asarray(result.concentrations[r])[measured]
        spectra = np.asarray(result.spectra[r])
        for s in range(dataset.n_species):
            ax_c.plot(
                time,
                projected[:, s],
                "o",
                ms=3,
                mec="none",
                alpha=0.5,
                color=color[s],
                label=label[s],
            )
            ax_c.plot(time, model[:, s], color=INK, lw=1)
            ax_s.plot(wavelength, spectra[s], color=color[s], lw=1.5, label=label[s])
        handles, names = ax_c.get_legend_handles_labels()
        handles += [
            Line2D([], [], ls="none", marker="o", ms=4, color=MUTED),
            Line2D([], [], color=INK, lw=1),
        ]
        ax_c.legend(handles, [*names, "projected spectra", "kinetic model"], frameon=False)
        ax_s.legend(frameon=False)

    if log_time:
        ax_c.set_xscale("log")
    ax_c.set(
        xlabel=f"time ({dataset.time_unit})", ylabel=f"concentration ({dataset.concentration_unit})"
    )
    ax_s.axhline(0, color=MUTED, lw=0.6)
    ax_s.set(
        xlabel=f"wavelength ({dataset.wavelength_unit})",
        ylabel=f"absorbance per {dataset.concentration_unit}",
    )
    _tag(ax_c, run)
    _tag(ax_s, run)
    return ax_c, ax_s


def plot_residuals(
    fit: KineticFit,
    dataset: SpectralDataset,
    run: str,
    *,
    axes: Sequence[Axes] | None = None,
) -> tuple[Axes, Axes]:
    """What the fit leaves in one run, in units of each wavelength's fitted noise level.

    Left, the residual map over wavelength and time, time running down, colour clipped at
    ±3: even static means the model and the noise level explain the data; stripes or
    blocks mean something is missing. Right, the root mean square of each spectrum's
    scaled residual. The noise level is fitted from the same residuals, so the rms over
    the run is 1 by construction; look for where it rises above 1.

    Args:
        fit: The fit of ``dataset``; a band has no single residual.
        dataset: The data that were fitted.
        run: The run's id, one of ``dataset.run_ids``.
        axes: Two axes, map then misfit; ``None`` makes a new figure.

    Returns:
        The map axes and the misfit axes.

    Raises:
        ValueError: If ``run`` is not a run of ``dataset``, or ``fit.run_ids`` differ from
            ``dataset.run_ids``.
    """
    r = dataset.run_index(run)
    _check_runs(fit, dataset)
    ax_map, ax_misfit = axes if axes is not None else _new_axes(2)

    measured = np.asarray(dataset.mask[r])
    time = np.asarray(dataset.time[r])[measured]
    scaled = np.asarray(fit.residuals[r])[measured] / np.asarray(fit.sigma[r])[None, :]
    mesh = ax_map.pcolormesh(
        np.asarray(dataset.wavelength),
        time,
        scaled,
        cmap=LinearSegmentedColormap.from_list("residual", DIVERGING),
        vmin=-3,
        vmax=3,
        shading="nearest",
        rasterized=True,
    )
    ax_map.invert_yaxis()
    ax_map.figure.colorbar(mesh, ax=ax_map, label="residual / noise level", ticks=(-3, 0, 3))
    ax_map.set(
        xlabel=f"wavelength ({dataset.wavelength_unit})", ylabel=f"time ({dataset.time_unit})"
    )

    misfit = np.sqrt((scaled**2).mean(axis=1))
    ax_misfit.plot(time, misfit, "o", ms=3, mec="none", color=MUTED)
    ax_misfit.axhline(1, color=INK, lw=1, label="noise level")
    ax_misfit.set(
        xlabel=f"time ({dataset.time_unit})", ylabel="rms residual / noise level", ylim=(0, None)
    )
    ax_misfit.legend(frameon=False)
    _tag(ax_map, run)
    _tag(ax_misfit, run)
    return ax_map, ax_misfit


def plot_rate_posterior(
    fit: KineticFit,
    condition: str,
    step: Step,
    *,
    ax: Axes | None = None,
) -> Axes:
    """The rate of one step in one condition: what each run claims, and the pooled result.

    Thin curves are each run's own posterior, normal on the log rate with
    ``fit.log_rate_sd`` (a diagnostic that is typically too narrow), labelled
    with the run id; a run whose ``log_rate_sd`` is ``NaN`` gets a dotted line at its rate.
    The thick curve is :meth:`~spectrahandler.kinetics.RateEstimate.density`, the
    replicate-based posterior behind the reported interval; it is left out with one run.

    Args:
        fit: A fit with the condition's replicate runs.
        condition: One of ``fit.conditions``.
        step: One of ``fit.scheme.steps``.
        ax: Axes to draw on; ``None`` makes a new figure.

    Returns:
        The axes drawn on.

    Raises:
        ValueError: If ``condition`` or ``step`` is not in the fit.
    """
    if condition not in fit.conditions:
        raise ValueError(
            f"no condition {condition!r}; the conditions are {sorted(set(fit.conditions))}"
        )
    if step not in fit.scheme.steps:
        raise ValueError(f"no step {step!r}; the steps are {list(fit.scheme.steps)}")
    ax = ax if ax is not None else _new_axes(1)[0]
    estimate = fit.summary()[condition].rates[step]
    j = fit.scheme.steps.index(step)
    runs = [i for i, c in enumerate(fit.conditions) if c == condition]
    log_k = np.log(np.asarray(fit.rates[runs, j]))
    sd = np.asarray(fit.log_rate_sd[runs, j])

    pooled = (
        None if estimate.between_sd_log is None else estimate.between_sd_log / np.sqrt(estimate.n)
    )
    widths = [w for w in (*sd[np.isfinite(sd)], pooled) if w is not None]
    spread = 5 * max(widths) if widths else 0.1
    grid = np.exp(np.linspace(log_k.min() - spread, log_k.max() + spread, 1000))
    for i, index in enumerate(runs):
        if np.isfinite(sd[i]):
            density = np.asarray(norm.pdf(jnp.log(grid), log_k[i], sd[i])) / grid
            ax.plot(grid, density, color=MUTED, lw=1)
            peak = int(density.argmax())
            where, coords = (grid[peak], density[peak]), "data"
        else:
            ax.axvline(np.exp(log_k[i]), color=MUTED, lw=1, ls=":")
            where, coords = (np.exp(log_k[i]), 1.0), ("data", "axes fraction")
        ax.annotate(
            fit.run_ids[index],
            where,
            xycoords=coords,
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize="small",
        )
    if pooled is not None:
        ax.plot(
            grid,
            np.asarray(estimate.density(grid)),
            color=INK,
            lw=2,
            label=f"pooled, {estimate.n} runs",
        )
        ax.legend(frameon=False)
    a, b = step
    ax.set(
        xlabel=f"rate {a} → {b} (1/{fit.time_unit})",
        ylabel=f"posterior density ({fit.time_unit})",
        ylim=(0, None),
    )
    return ax
