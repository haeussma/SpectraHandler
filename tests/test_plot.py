"""The noise figure renders. What it shows is pinned by test_diagnostics.py."""

import jax
import matplotlib

matplotlib.use("Agg")

import pytest

from spectrahandler.curve_resolution import make_realistic_dataset, noise_diagnostics
from spectrahandler.kinetics import Scheme, fit_kinetics, make_kinetic_replicates
from spectrahandler.plot import plot_kinetic_fit, plot_noise


def test_plot_noise_has_three_panels() -> None:
    data, _, _ = make_realistic_dataset(jax.random.key(0))
    fig = plot_noise(noise_diagnostics(data), data)
    titles = [ax.get_title(loc="left") for ax in fig.axes if ax.get_title(loc="left")]
    assert len(titles) == 3
    assert titles[0].startswith("what the data hold")


def test_plot_kinetic_fit_has_three_panels() -> None:
    scheme = Scheme(steps=[("A", "B")])
    data, _, _ = make_kinetic_replicates(
        jax.random.key(0), scheme, {("A", "B"): 0.8}, initial={"A": 10.0}
    )
    fig = plot_kinetic_fit(fit_kinetics(data, scheme), data, wavelengths=[400, 520])
    titles = [ax.get_title(loc="left") for ax in fig.axes if ax.get_title(loc="left")]
    assert len(titles) == 3
    assert titles[0].startswith("data and model")


def test_plot_kinetic_fit_rejects_four_wavelengths() -> None:
    scheme = Scheme(steps=[("A", "B")])
    data, _, _ = make_kinetic_replicates(
        jax.random.key(0), scheme, {("A", "B"): 0.8}, initial={"A": 10.0}, n_replicates=1
    )
    with pytest.raises(ValueError, match="1 to 3"):
        plot_kinetic_fit(fit_kinetics(data, scheme), data, wavelengths=[400, 450, 500, 550])
