"""Pooling replicates: the t quantile, coverage of the interval, and the warnings."""

import jax
import numpy as np
import pytest

from spectrahandler.kinetics import Scheme, fit_kinetics, make_kinetic_replicates
from spectrahandler.kinetics.summary import student_t_quantile

ONE_STEP = Scheme(steps=[("A", "B")])
STEP = ("A", "B")


@pytest.mark.parametrize(
    ("p", "df", "expected"),
    [
        (0.975, 1.0, 12.706204736),
        (0.975, 3.0, 3.182446305),
        (0.975, 30.0, 2.042272456),
        (0.95, 10.0, 1.812461123),
    ],
)
def test_student_t_quantile(p: float, df: float, expected: float) -> None:
    assert student_t_quantile(p, df) == pytest.approx(expected, rel=1e-8)


def test_interval_covers_the_true_rate_at_the_stated_level() -> None:
    """300 conditions x 4 replicates; replicate log rates scatter by 2 %."""
    hits = []
    for i in range(300):
        data, _, _ = make_kinetic_replicates(
            jax.random.key(i), ONE_STEP, {STEP: 0.8}, initial={"A": 10.0}, between_sd_log=0.02
        )
        estimate = fit_kinetics(data, ONE_STEP).summary()["synthetic"].rates[STEP]
        assert estimate.lower is not None
        assert estimate.upper is not None
        hits.append(estimate.lower <= 0.8 <= estimate.upper)
    assert 0.92 <= np.mean(hits) <= 0.98


def test_conditions_are_pooled_separately(key: jax.Array) -> None:
    data, _, _ = make_kinetic_replicates(
        key, ONE_STEP, {STEP: 0.8}, initial={"A": 10.0}, between_sd_log=0.02
    )
    from spectrahandler.curve_resolution import SpectralDataset

    relabelled = SpectralDataset.create(
        absorbance=data.absorbance,
        time=data.time,
        wavelength=data.wavelength,
        species=data.species,
        initial_state=data.initial_state,
        run_ids=data.run_ids,
        conditions=("low", "low", "high", "high"),
        time_unit=data.time_unit,
    )
    summary = fit_kinetics(relabelled, ONE_STEP).summary()
    assert list(summary) == ["low", "high"]
    assert summary["high"].run_ids == ("replicate3", "replicate4")
    assert summary["low"].rates[STEP].n == 2


def test_one_run_has_no_interval(key: jax.Array) -> None:
    data, _, _ = make_kinetic_replicates(
        key, ONE_STEP, {STEP: 0.8}, initial={"A": 10.0}, n_replicates=1
    )
    summary = fit_kinetics(data, ONE_STEP).summary()["synthetic"]
    assert summary.rates[STEP].lower is None
    assert summary.spectrum_lower is None
    assert any("no replicate-based uncertainty" in w for w in summary.warnings)


def test_warns_when_replicates_do_not_spread_beyond_fit_precision(key: jax.Array) -> None:
    data, _, _ = make_kinetic_replicates(
        key, ONE_STEP, {STEP: 0.8}, initial={"A": 10.0}, between_sd_log=0.0
    )
    summary = fit_kinetics(data, ONE_STEP).summary()["synthetic"]
    assert any("not clearly larger" in w for w in summary.warnings)


def test_no_warning_when_replicates_spread(key: jax.Array) -> None:
    data, _, _ = make_kinetic_replicates(
        key, ONE_STEP, {STEP: 0.8}, initial={"A": 10.0}, between_sd_log=0.05
    )
    assert fit_kinetics(data, ONE_STEP).summary()["synthetic"].warnings == ()


def test_spectrum_band_brackets_the_mean(key: jax.Array) -> None:
    data, _, spectra = make_kinetic_replicates(
        key, ONE_STEP, {STEP: 0.8}, initial={"A": 10.0}, between_sd_log=0.02
    )
    summary = fit_kinetics(data, ONE_STEP).summary()["synthetic"]
    assert summary.spectrum_lower is not None
    assert summary.spectrum_upper is not None
    assert bool((summary.spectrum_lower <= summary.spectrum_mean).all())
    assert bool((summary.spectrum_mean <= summary.spectrum_upper).all())
    np.testing.assert_allclose(summary.spectrum_mean, spectra, atol=5e-4)


def test_swap_ambiguity_becomes_a_warning(key: jax.Array) -> None:
    two = Scheme(steps=[("A", "B"), ("B", "C")])
    data, _, _ = make_kinetic_replicates(
        key, two, {("A", "B"): 0.6, ("B", "C"): 0.25}, initial={"A": 10.0}
    )
    summary = fit_kinetics(data, two, initial_rates={("A", "B"): 0.5, ("B", "C"): 0.2}).summary()[
        "synthetic"
    ]
    assert any("not separately identified" in w for w in summary.warnings)
