"""The figures render, carry no titles, and draw the numbers they claim to draw."""

import dataclasses

import jax
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PolyCollection, QuadMesh

from spectrahandler.curve_resolution import (
    SpectralDataset,
    make_realistic_dataset,
    noise_diagnostics,
)
from spectrahandler.curve_resolution.band import FeasibleBand, resolve_band
from spectrahandler.kinetics import KineticFit, Scheme, fit_kinetics, make_kinetic_replicates
from spectrahandler.plot import (
    plot_deconvolution,
    plot_noise,
    plot_rate_posterior,
    plot_residuals,
    plot_spectra,
)

matplotlib.use("Agg")

STEP = ("A", "B")


def _no_titles(axes: list[Axes]) -> bool:
    return all(ax.get_title(loc=loc) == "" for ax in axes for loc in ("left", "center", "right"))


def test_plot_noise_has_three_panels_and_no_titles() -> None:
    data, _, _ = make_realistic_dataset(jax.random.key(0))
    fig = plot_noise(noise_diagnostics(data), data)
    assert len(fig.axes) == 4, "three panels and a colorbar"
    assert _no_titles(fig.axes)
    labels = fig.axes[0].get_legend_handles_labels()[1]
    assert any(label.startswith("rank ") for label in labels)


@pytest.fixture(scope="module")
def kinetic() -> tuple[SpectralDataset, KineticFit]:
    scheme = Scheme(steps=[STEP])
    data, _, _ = make_kinetic_replicates(
        jax.random.key(0), scheme, {STEP: 0.8}, initial={"A": 10.0}, between_sd_log=0.02
    )
    return data, fit_kinetics(data, scheme)


def _tags(ax: Axes) -> list[str]:
    return [text.get_text() for text in ax.texts]


def test_plot_spectra_draws_every_measured_spectrum(
    kinetic: tuple[SpectralDataset, KineticFit],
) -> None:
    data, _ = kinetic
    ax = plot_spectra(data, "replicate2")
    (lines,) = [c for c in ax.collections if isinstance(c, LineCollection)]
    assert len(lines.get_segments()) == int(data.mask[1].sum())
    assert len(ax.figure.axes) == 2, "panel and time colorbar"
    assert _tags(ax) == ["replicate2"]
    assert _no_titles(ax.figure.axes)


def test_unknown_run_is_an_error(kinetic: tuple[SpectralDataset, KineticFit]) -> None:
    data, _ = kinetic
    with pytest.raises(ValueError, match="no run 'shot9'"):
        plot_spectra(data, "shot9")


@pytest.fixture(scope="module")
def band() -> tuple[SpectralDataset, FeasibleBand]:
    data, _, _ = make_realistic_dataset(jax.random.key(0))
    return data, resolve_band(data, jax.random.key(1), n_iter=200, n_burn=100)


def _legend(ax: Axes) -> list[str]:
    legend = ax.get_legend()
    assert legend is not None
    return [text.get_text() for text in legend.get_texts()]


def test_panels_draw_on_the_axes_they_are_given(
    kinetic: tuple[SpectralDataset, KineticFit],
) -> None:
    data, fit = kinetic
    fig, axes = plt.subplots(1, 2)
    assert plot_deconvolution(fit, data, "replicate1", axes=axes) == tuple(axes)
    assert plot_spectra(data, "replicate1", ax=axes[0]) is axes[0]
    plt.close(fig)


def test_deconvolution_of_a_fit_draws_projection_and_model(
    kinetic: tuple[SpectralDataset, KineticFit],
) -> None:
    data, fit = kinetic
    ax_c, ax_s = plot_deconvolution(fit, data, "replicate3", labels={"A": "start"})
    dots = [line for line in ax_c.lines if line.get_marker() == "o"]
    model = [line for line in ax_c.lines if line.get_marker() != "o"]
    projected = np.asarray(data.project(fit.spectra))[2]
    for s in range(data.n_species):
        np.testing.assert_allclose(np.asarray(dots[s].get_ydata()), projected[:, s])
        np.testing.assert_allclose(
            np.asarray(model[s].get_ydata()), np.asarray(fit.concentrations)[2, :, s]
        )
    assert _legend(ax_c) == ["start", "B", "projected spectra", "kinetic model"]
    assert _tags(ax_c) == _tags(ax_s) == ["replicate3"]
    assert _no_titles(ax_c.figure.axes)


def test_deconvolution_of_a_band_shades_every_species(
    band: tuple[SpectralDataset, FeasibleBand],
) -> None:
    data, result = band
    ax_c, ax_s = plot_deconvolution(result, data, data.run_ids[0])
    for ax in (ax_c, ax_s):
        shaded = [c for c in ax.collections if isinstance(c, PolyCollection)]
        assert len(shaded) == data.n_species


def test_colors_must_cover_every_species(kinetic: tuple[SpectralDataset, KineticFit]) -> None:
    data, fit = kinetic
    with pytest.raises(ValueError, match="colors has no entry for"):
        plot_deconvolution(fit, data, "replicate1", colors={"A": "red"})


def test_residuals_are_scaled_by_the_noise_level(
    kinetic: tuple[SpectralDataset, KineticFit],
) -> None:
    data, fit = kinetic
    ax_map, ax_misfit = plot_residuals(fit, data, "replicate1")
    (mesh,) = [c for c in ax_map.collections if isinstance(c, QuadMesh)]
    scaled = np.asarray(fit.residuals[0] / fit.sigma[0][None, :])
    np.testing.assert_allclose(np.asarray(mesh.get_array()).reshape(scaled.shape), scaled)
    misfit = np.asarray(ax_misfit.lines[0].get_ydata())
    np.testing.assert_allclose(misfit, np.sqrt((scaled**2).mean(axis=1)))
    assert float(np.sqrt((misfit**2).mean())) == pytest.approx(1.0), "sigma is fitted from these"
    assert _tags(ax_map) == _tags(ax_misfit) == ["replicate1"]
    assert _no_titles(ax_map.figure.axes)


def test_a_fit_of_other_runs_is_refused(kinetic: tuple[SpectralDataset, KineticFit]) -> None:
    data, fit = kinetic
    reversed_fit = dataclasses.replace(fit, run_ids=fit.run_ids[::-1])
    with pytest.raises(ValueError, match="run_ids"):
        plot_residuals(reversed_fit, data, "replicate1")
    with pytest.raises(ValueError, match="run_ids"):
        plot_deconvolution(reversed_fit, data, "replicate1")


def test_rate_posterior_labels_every_run(kinetic: tuple[SpectralDataset, KineticFit]) -> None:
    data, fit = kinetic
    ax = plot_rate_posterior(fit, "synthetic", STEP)
    assert sorted(text.get_text() for text in ax.texts) == sorted(data.run_ids)
    assert len(ax.lines) == data.n_run + 1, "one curve per run and the pooled one"
    assert _legend(ax) == ["pooled, 4 runs"]
    estimate = fit.summary()["synthetic"].rates[STEP]
    (pooled,) = [line for line in ax.lines if line.get_label() == "pooled, 4 runs"]
    xdata = np.asarray(pooled.get_xdata())
    np.testing.assert_allclose(np.asarray(pooled.get_ydata()), np.asarray(estimate.density(xdata)))
    with pytest.raises(ValueError, match="no step"):
        plot_rate_posterior(fit, "synthetic", ("B", "A"))


def test_rate_posterior_marks_a_run_without_curvature(
    kinetic: tuple[SpectralDataset, KineticFit],
) -> None:
    data, fit = kinetic
    flat = dataclasses.replace(fit, log_rate_sd=fit.log_rate_sd.at[0, 0].set(np.nan))
    ax = plot_rate_posterior(flat, "synthetic", STEP)
    assert sorted(text.get_text() for text in ax.texts) == sorted(data.run_ids)
    dotted = [line for line in ax.lines if line.get_linestyle() == ":"]
    assert len(dotted) == 1
