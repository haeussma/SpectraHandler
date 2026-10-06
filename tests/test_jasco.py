"""The JASCO readers against the committed fixtures and their documented traps."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from spectrahandler.jasco import read_interval_scan, read_spectrum, read_spectrum_series

PROBE_A = "probe_a/20260826_Probe_a.csv"
PROBE_C = "probe_a/20260826_Probe_c_titancitrate-only.csv"
BUFFER = "probe_a/20260827_Puffer_background.csv"


def test_interval_scan_shape_and_order(data_dir: Path) -> None:
    scan = read_interval_scan(data_dir / PROBE_A)
    assert scan.absorbance.shape == (60, 551)
    assert scan.wavelength_nm[0] == 250.0
    assert scan.wavelength_nm[-1] == 800.0
    assert bool(np.all(np.diff(scan.wavelength_nm) > 0))


def test_interval_scan_values_follow_their_wavelength(data_dir: Path) -> None:
    """File row ``800,0.0172124,...``: A(800 nm, first time) after sorting and transposing."""
    scan = read_interval_scan(data_dir / PROBE_A)
    assert scan.absorbance[0, -1] == pytest.approx(0.0172124)


def test_interval_scan_keeps_recorded_minutes(data_dir: Path) -> None:
    """Probe c starts 2.783 min after Probe a; never rebuilt as a uniform grid."""
    scan = read_interval_scan(data_dir / PROBE_C)
    assert scan.time_min[0] == pytest.approx(2.78333)
    assert 570 < scan.time_min[-1] < 600


def test_header_is_kept(data_dir: Path) -> None:
    assert read_interval_scan(data_dir / PROBE_A).metadata["YUNITS"] == "ABSORBANCE"


def test_single_spectrum_stops_before_footer(data_dir: Path) -> None:
    """The buffer reads about -0.194 AU at 250 nm: the deep-UV instrument artefact."""
    wavelength_nm, absorbance, header = read_spectrum(data_dir / BUFFER)
    assert wavelength_nm.shape == absorbance.shape
    assert wavelength_nm[0] == 250.0
    assert absorbance[0] == pytest.approx(-0.194, abs=0.001)
    assert "TITLE" in header


def test_series_is_ordered_by_time_not_name(data_dir: Path) -> None:
    """Lexical order would put 15min before 1h before 20h before 2h."""
    scan = read_spectrum_series(sorted((data_dir / "1a").glob("*.csv")))
    np.testing.assert_array_equal(scan.time_min, [0, 15, 30, 60, 120, 240, 1200])
    assert scan.absorbance.shape == (7, 801)
    assert scan.wavelength_nm[0] == 300.0


def test_series_shows_saturation(data_dir: Path) -> None:
    """``1a`` saturates at 10 AU; the reader must not hide it."""
    scan = read_spectrum_series(sorted((data_dir / "1a").glob("*.csv")))
    assert float(scan.absorbance.max()) == pytest.approx(10.0)


def test_missing_marker_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "not_jasco.csv"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="XYDATA"):
        read_interval_scan(path)


def test_series_needs_a_time_in_every_name(data_dir: Path) -> None:
    with pytest.raises(ValueError, match="acquisition time"):
        read_spectrum_series([data_dir / BUFFER])


def test_series_rejects_two_files_at_one_time(data_dir: Path, tmp_path: Path) -> None:
    shutil.copy(data_dir / "1a" / "1a_1h.csv", tmp_path / "1a_1h.csv")
    shutil.copy(data_dir / "1a" / "1a_1h.csv", tmp_path / "1a_60min.csv")
    with pytest.raises(ValueError, match="same acquisition time"):
        read_spectrum_series(sorted(tmp_path.glob("*.csv")))


def test_blank_line_inside_data_is_rejected(data_dir: Path, tmp_path: Path) -> None:
    """A stray blank line would end the block early; NPOINTS catches the truncation."""
    lines = (data_dir / PROBE_A).read_text().splitlines()
    marker = next(i for i, ln in enumerate(lines) if ln.strip() == "XYDATA")
    lines.insert(marker + 101, "")
    path = tmp_path / "truncated.csv"
    path.write_text("\n".join(lines))
    with pytest.raises(ValueError, match=r"truncated.csv: 99 data rows .* NPOINTS says 551"):
        read_interval_scan(path)


def test_single_spectrum_row_count_must_match_npoints(tmp_path: Path) -> None:
    path = tmp_path / "short.csv"
    path.write_text("NPOINTS,     3\nXYDATA\n700,0.1\n699,0.2\n")
    with pytest.raises(ValueError, match=r"2 data rows .* NPOINTS says 3"):
        read_spectrum(path)


def test_empty_xydata_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("TITLE,x\nXYDATA\n\nfooter,1\n")
    with pytest.raises(ValueError, match="no data rows"):
        read_interval_scan(path)
    with pytest.raises(ValueError, match="no data rows"):
        read_spectrum(path)


def test_empty_series_is_rejected() -> None:
    with pytest.raises(ValueError, match="no files"):
        read_spectrum_series([])
