"""Regression against local-only data. Skips unless the expectations file exists."""

import json
from pathlib import Path

import numpy as np
import pytest

from spectrahandler.curve_resolution import SpectralDataset
from spectrahandler.kinetic_studio import read_kinetic_studio
from spectrahandler.kinetics import Scheme, fit_kinetics

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = ROOT / "local" / "regression" / "stopped_flow_expected.json"

pytestmark = pytest.mark.skipif(
    not EXPECTED.exists(), reason="local regression data is not committed"
)


def test_local_regression_matches_the_expected_values() -> None:
    spec = json.loads(EXPECTED.read_text(encoding="utf-8"))
    scans = [read_kinetic_studio(ROOT / spec["data_dir"] / f"{n}.csv") for n in spec["files"]]
    low, high = spec["wavelength_nm"]
    keep = (scans[0].wavelength_nm >= low) & (scans[0].wavelength_nm <= high)
    data = SpectralDataset.from_runs(
        [(s.time_s, s.absorbance[:, keep]) for s in scans],
        wavelength=scans[0].wavelength_nm[keep],
        species=tuple(spec["species"]),
        initial_state=spec["initial_state"],
        run_ids=spec["run_ids"],
        conditions=[spec["condition"]] * len(scans),
        time_unit="s",
    )
    step = tuple(spec["step"])
    fit = fit_kinetics(data, Scheme(steps=[step]))
    atol = spec["atol"]
    np.testing.assert_allclose(fit.rates[:, 0], spec["rates"], atol=atol)
    rate = fit.summary()[spec["condition"]].rates[step]
    assert rate.value == pytest.approx(spec["geometric_mean"], abs=atol)
    assert rate.lower == pytest.approx(spec["lower"], abs=atol)
    assert rate.upper == pytest.approx(spec["upper"], abs=atol)
