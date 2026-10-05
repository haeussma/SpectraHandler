"""The noise figure renders. What it shows is pinned by test_diagnostics.py."""

import jax
import matplotlib

matplotlib.use("Agg")

from spectrahandler.curve_resolution import make_realistic_dataset, noise_diagnostics
from spectrahandler.plot import plot_noise


def test_plot_noise_has_three_panels() -> None:
    data, _, _ = make_realistic_dataset(jax.random.key(0))
    fig = plot_noise(noise_diagnostics(data), data)
    titles = [ax.get_title(loc="left") for ax in fig.axes if ax.get_title(loc="left")]
    assert len(titles) == 3
    assert titles[0].startswith("what the data hold")
