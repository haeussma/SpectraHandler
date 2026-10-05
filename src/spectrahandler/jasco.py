"""Readers for JASCO V-730 CSV exports.

Two layouts, both documented with their traps in ``tests/data/README.md``: an interval
scan (one file, every timepoint) and a single spectrum (one file per timepoint, time in
the filename). Readers return the file's numbers unchanged apart from sorting wavelength
ascending; no trimming, blanking or resampling happens here.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

__all__ = ["Scan", "read_interval_scan", "read_spectrum", "read_spectrum_series"]

type _Floats = npt.NDArray[np.float64]

#: Filename suffix such as ``_15min`` or ``_2h``. Units are mixed within one series.
_TIME_IN_NAME = re.compile(r"_(\d+(?:\.\d+)?)(min|h)$")
_MINUTES_PER = {"min": 1.0, "h": 60.0}


@dataclass(frozen=True)
class Scan:
    """One run as recorded: absorbance over time and wavelength.

    Attributes:
        wavelength_nm: Strictly increasing grid, shape ``(n_wavelength,)``, in nm.
        time_min: Acquisition times as recorded, shape ``(n_time,)``, in minutes. Not
            rebuilt as a uniform grid; runs from one cell changer do not share an origin.
        absorbance: Shape ``(n_time, n_wavelength)``, in absorbance units.
        metadata: The ``KEY,value`` header of the (first) file, values stripped.
    """

    wavelength_nm: _Floats
    time_min: _Floats
    absorbance: _Floats
    metadata: dict[str, str]


def _split(path: Path) -> tuple[dict[str, str], list[list[str]]]:
    """Header as a dict, and the data rows after ``XYDATA`` up to the first blank line."""
    lines = path.read_text(errors="replace").splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip().upper() == "XYDATA")
    except StopIteration:
        raise ValueError(f"{path}: no XYDATA marker; not a JASCO CSV export") from None
    header = {}
    for line in lines[:start]:
        key, _, value = line.partition(",")
        header[key.strip()] = value.strip()
    rows = []
    for line in lines[start + 1 :]:
        if not line.strip():
            break
        rows.append(line.strip().split(","))
    return header, rows


def _ascending(wavelength_nm: _Floats, absorbance: _Floats) -> tuple[_Floats, _Floats]:
    """Sort the wavelength axis (last axis of ``absorbance``) ascending; JASCO counts down."""
    order = np.argsort(wavelength_nm)
    wavelength_nm, absorbance = wavelength_nm[order], absorbance[..., order]
    if not np.all(np.diff(wavelength_nm) > 0):
        raise ValueError("duplicate wavelengths in export")
    return wavelength_nm, absorbance


def read_interval_scan(path: str | Path) -> Scan:
    """Read a JASCO interval-scan export: every timepoint of one run in one file.

    The first data row holds the acquisition times **in minutes** behind an empty first
    field; every later row is ``wavelength, A(t0), A(t1), ...``.

    Args:
        path: The ``.csv`` export.

    Returns:
        The run, wavelength ascending, absorbance as ``(n_time, n_wavelength)``.

    Raises:
        ValueError: If the file has no ``XYDATA`` marker or a row has the wrong length.
    """
    path = Path(path)
    header, rows = _split(path)
    time_min = np.array([float(v) for v in rows[0][1:] if v.strip()])
    body = rows[1:]
    if any(len(r) != len(time_min) + 1 for r in body):
        raise ValueError(f"{path}: a data row does not have one value per timepoint")
    table = np.array(body, dtype=np.float64)
    wavelength_nm, absorbance = _ascending(table[:, 0], table[:, 1:].T)
    return Scan(wavelength_nm, time_min, absorbance, header)


def _time_from_name(path: Path) -> float:
    """Acquisition time in minutes from a name such as ``1a_15min`` or ``1a_2h``."""
    match = _TIME_IN_NAME.search(path.stem)
    if match is None:
        raise ValueError(f"{path.name}: no acquisition time like '_15min' or '_2h' in name")
    return float(match.group(1)) * _MINUTES_PER[match.group(2)]


def read_spectrum(path: str | Path) -> tuple[_Floats, _Floats, dict[str, str]]:
    """Read one JASCO single-spectrum export, e.g. a blank or a reference scan.

    Two columns, ``wavelength,absorbance``, then an instrument footer after a blank line.

    Args:
        path: The ``.csv`` export.

    Returns:
        Wavelength in nm ascending with shape ``(n_wavelength,)``, absorbance with the
        same shape, and the ``KEY,value`` header.

    Raises:
        ValueError: If the file has no ``XYDATA`` marker or the rows are not two columns.
    """
    path = Path(path)
    header, rows = _split(path)
    if any(len(r) != 2 for r in rows):
        raise ValueError(f"{path}: expected two columns, wavelength and absorbance")
    table = np.array(rows, dtype=np.float64)
    wavelength_nm, absorbance = _ascending(table[:, 0], table[:, 1])
    return wavelength_nm, absorbance, header


def read_spectrum_series(paths: Sequence[str | Path]) -> Scan:
    """Read single-spectrum exports into one run, ordered by the time in each filename.

    Each file is two columns, ``wavelength,absorbance``, followed by an instrument
    footer after a blank line. The acquisition time is only in the filename, with mixed
    units (``_15min``, ``_1h``); files are ordered by that time, never by name.

    Args:
        paths: One export per timepoint, in any order. All on the same wavelength grid.

    Returns:
        The run, wavelength ascending, with ``metadata`` from the earliest file.

    Raises:
        ValueError: If a filename carries no time, two files share a time, or the
            wavelength grids differ.
    """
    files = sorted((Path(p) for p in paths), key=_time_from_name)
    time_min = np.array([_time_from_name(p) for p in files])
    if len(set(time_min.tolist())) != len(time_min):
        raise ValueError("two files carry the same acquisition time")
    spectra = [read_spectrum(p) for p in files]
    grid = spectra[0][0]
    if any(w.shape != grid.shape or not np.array_equal(w, grid) for w, _, _ in spectra):
        raise ValueError("spectra are not on one wavelength grid")
    absorbance = np.stack([a for _, a, _ in spectra])
    return Scan(grid, time_min, absorbance, spectra[0][2])
