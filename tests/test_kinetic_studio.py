"""The Kinetic Studio reader against a small file in the instrument's layout."""

from pathlib import Path

import numpy as np
import pytest

from spectrahandler.kinetic_studio import read_kinetic_studio

_FILE = (
    'File Info:"," DX2 Kinetics File:example.ksd | 01/01/2025 12:00:00" Temp: 25.0,\n'
    ",0,0.01,0.02,\n"
    "301.0,0.30,0.31,0.32,\n"
    "300.5,0.20,,0.22,\n"
    "\n"
    "end,\n"
)


@pytest.fixture
def export(tmp_path: Path) -> Path:
    path = tmp_path / "shot.csv"
    path.write_text(_FILE, encoding="utf-8")
    return path


def test_shape_order_and_values(export: Path) -> None:
    scan = read_kinetic_studio(export)
    np.testing.assert_array_equal(scan.time_s, [0.0, 0.01, 0.02])
    np.testing.assert_array_equal(scan.wavelength_nm, [300.5, 301.0])
    assert scan.absorbance.shape == (3, 2)
    assert scan.absorbance[2, 1] == pytest.approx(0.32)
    assert scan.absorbance[0, 0] == pytest.approx(0.20)


def test_empty_cell_is_nan_not_dropped(export: Path) -> None:
    scan = read_kinetic_studio(export)
    assert np.isnan(scan.absorbance[1, 0])


def test_header_fields(export: Path) -> None:
    header = read_kinetic_studio(export).header
    assert header["file"] == "example.ksd"
    assert header["timestamp"] == "01/01/2025 12:00:00"
    assert header["temperature"] == "25.0"
    assert header["raw"].startswith("File Info")


def test_rejects_repeated_wavelength(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("x\n,0,1\n300,1,2\n300,1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="more than once"):
        read_kinetic_studio(path)


def test_rejects_file_without_data(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("x\n,0,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected a header line"):
        read_kinetic_studio(path)


def test_rejects_row_with_more_values_than_timepoints(tmp_path: Path) -> None:
    path = tmp_path / "wide.csv"
    path.write_text("x\n,0,1,\n300,1,2,\n301,1,2,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="row 301 has more values than timepoints"):
        read_kinetic_studio(path)
