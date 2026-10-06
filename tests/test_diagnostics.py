"""Rank and noise diagnostics: do they tell noise from structure?"""

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from spectrahandler.curve_resolution import (
    SpectralDataset,
    make_realistic_dataset,
    noise_diagnostics,
)
from spectrahandler.jasco import read_interval_scan


def _with_absorbance(dataset: SpectralDataset, absorbance: jax.Array) -> SpectralDataset:
    return SpectralDataset.create(
        absorbance=absorbance,
        time=dataset.time,
        wavelength=dataset.wavelength,
        species=dataset.species,
        initial_state=dataset.initial_state,
        run_ids=dataset.run_ids,
    )


def test_white_noise_reads_as_three_components() -> None:
    """Measured: 3 above the edge, sigma 1.98e-3, lag-1 autocorrelation -0.07 / -0.09."""
    data, _, _ = make_realistic_dataset(jax.random.key(0))
    diag = noise_diagnostics(data)
    assert diag.rank == 3
    assert diag.n_above_noise == 3
    assert diag.sigma == pytest.approx(0.002, rel=0.1)
    assert diag.sigma == pytest.approx(float(diag.sigma_by_rank[3]))
    assert float(diag.autocorrelation_wavelength[0]) == pytest.approx(1.0)
    assert abs(float(diag.autocorrelation_wavelength[1])) < 0.15
    assert abs(float(diag.autocorrelation_time[1])) < 0.15
    assert diag.residual.shape == data.absorbance.shape


def test_an_unmodelled_offset_shows_as_a_fourth_component() -> None:
    """A flat offset drifting over time is one more component. Measured: 4 above."""
    data, _, _ = make_realistic_dataset(jax.random.key(0))
    drift = 0.01 * jnp.sin(jnp.linspace(0.0, 3.0, data.n_time))[None, :, None]
    diag = noise_diagnostics(_with_absorbance(data, data.absorbance + drift))
    assert diag.n_above_noise == 4


def test_interpolated_export_is_not_white(data_dir: Path) -> None:
    """The trap in tests/data/README.md: the JASCO export is interpolated along wavelength.

    Measured on Probe a (>= 340 nm, rank 6): lag-1 autocorrelation 0.96 along wavelength.
    White noise would read about 0, so the rank-residual sigma cannot be trusted here.
    """
    scan = read_interval_scan(data_dir / "probe_a" / "20260826_Probe_a.csv")
    keep = scan.wavelength_nm >= 340
    data = SpectralDataset.create(
        absorbance=jnp.asarray(scan.absorbance[None][..., keep]),
        time=jnp.asarray(scan.time_min[None] / 60),
        wavelength=jnp.asarray(scan.wavelength_nm[keep]),
        species=("cob1", "cob2", "co3", "mecbl", "ti3", "tiox"),
        initial_state=jnp.zeros((1, 6)),
        run_ids=("a",),
    )
    diag = noise_diagnostics(data)
    assert float(diag.autocorrelation_wavelength[1]) > 0.9


def test_rank_must_leave_room_for_noise() -> None:
    data, _, _ = make_realistic_dataset(jax.random.key(0))
    with pytest.raises(ValueError, match="rank"):
        noise_diagnostics(data, rank=30)


def test_requires_fully_measured_runs() -> None:
    data, _, _ = make_realistic_dataset(jax.random.key(0))
    masked = SpectralDataset.create(
        absorbance=data.absorbance,
        time=data.time,
        wavelength=data.wavelength,
        species=data.species,
        initial_state=data.initial_state,
        run_ids=data.run_ids,
        mask=data.mask.at[0, 3].set(False),
    )
    with pytest.raises(NotImplementedError, match="fully measured"):
        noise_diagnostics(masked)
