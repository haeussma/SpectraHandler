"""Reader for TgK Scientific Kinetic Studio CSV exports (stopped-flow, diode array).

Layout, from a KinetAsyst SF-61DX2 with a DET2B diode array::

    File Info:"," DX2 Kinetics File:example.ksd | 01/01/2025 12:00:00" Temp: 25.0,
    ,0,0.01,0.02, ... ,0.99          <- leading empty field, then times in seconds
    300.21,0.21666,0.18396, ...      <- one row per wavelength

The reader returns the file's numbers unchanged apart from sorting wavelength ascending.
Traps it does not fix:

- The ``_NNNs`` in a filename is the acquisition window, not a time point.
- Times are returned as recorded. Some exports label each frame one sampling step late for
  longer acquisition windows; the label is not the reaction time.
- The temperature in the header is kept as text: the instrument's reading is unreliable.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

__all__ = ["KineticStudioScan", "read_kinetic_studio"]

type _Floats = npt.NDArray[np.float64]

_HEADER = re.compile(
    r'File:\s*(?P<file>[^|"]+?)\s*\|\s*(?P<timestamp>[^"]+?)\s*"\s*Temp:\s*(?P<temperature>[^,]*)'
)


@dataclass(frozen=True, eq=False)
class KineticStudioScan:
    """One stopped-flow shot as recorded.

    Attributes:
        wavelength_nm: Strictly increasing grid, shape ``(n_wavelength,)``, in nm.
        time_s: Frame times as recorded, shape ``(n_time,)``, in seconds.
        absorbance: Shape ``(n_time, n_wavelength)``, in absorbance units. An empty cell
            in the file is ``NaN``, never dropped.
        header: ``raw`` (the first line), and ``file``, ``timestamp`` and
            ``temperature`` (text) when the first line has the usual form.
    """

    wavelength_nm: _Floats
    time_s: _Floats
    absorbance: _Floats
    header: dict[str, str]


def _number(field: str) -> float:
    """A cell as a float; an empty cell is ``NaN``."""
    field = field.strip()
    return float(field) if field else float("nan")


def read_kinetic_studio(path: str | Path) -> KineticStudioScan:
    """Read one Kinetic Studio CSV export.

    Rows with fewer than three fields (blank lines, footers) are skipped: they are not
    spectra. Single-wavelength trace files are not supported.

    Args:
        path: The ``.csv`` export.

    Returns:
        The shot, with wavelength sorted ascending.

    Raises:
        ValueError: If the file has no time row or no data rows, repeats a wavelength, or has
            a data row with more values than there are timepoints.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 3:
        raise ValueError(f"{path}: expected a header line, a time row and data rows")
    header = {"raw": lines[0]}
    match = _HEADER.search(lines[0])
    if match:
        header.update({k: v.strip() for k, v in match.groupdict().items()})

    time_fields = lines[1].split(",")[1:]
    while time_fields and not time_fields[-1].strip():  # trailing comma, not a timepoint
        time_fields.pop()
    time_s = np.array([float(f) for f in time_fields])

    rows = [line.split(",") for line in lines[2:]]
    rows = [row for row in rows if len(row) >= 3]
    if not rows or time_s.size == 0:
        raise ValueError(f"{path}: no time row or no data rows")
    wavelength = np.array([float(row[0]) for row in rows])
    absorbance = np.full((len(rows), time_s.size), np.nan)
    for i, row in enumerate(rows):  # IO boundary: one row per wavelength
        if any(f.strip() for f in row[1 + time_s.size :]):
            raise ValueError(f"{path}: row {row[0]} has more values than timepoints")
        cells = [_number(f) for f in row[1 : 1 + time_s.size]]
        absorbance[i, : len(cells)] = cells

    order = np.argsort(wavelength, kind="stable")
    if np.any(np.diff(wavelength[order]) <= 0):
        raise ValueError(f"{path}: a wavelength appears more than once")
    return KineticStudioScan(
        wavelength_nm=wavelength[order],
        time_s=time_s,
        absorbance=np.ascontiguousarray(absorbance[order].T),
        header=header,
    )
