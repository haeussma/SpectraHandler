"""The raw fixtures still have the shape `tests/data/README.md` documents.

No SpectraHandler code is exercised here on purpose -- these assertions are what a
reader must cope with, and they fail if someone re-exports the data differently.
"""

from itertools import pairwise
from pathlib import Path

import pytest

#: filename -> first acquisition time in minutes. The cell changer visits the cuvettes in
#: sequence, so the runs share a clock but not an origin. Aligning them on index is wrong.
PROBES = {
    "20260826_Probe_a.csv": 0.0,
    "20260826_Probe_c_titancitrate-only.csv": 2.78333,
    "20260826_Probe_d_hydroxycobalamin+titancitrate.csv": 4.18333,
    "20260826_Probe_f_only_enzyme.csv": 6.98333,
}


def _data_block(path: Path) -> list[str]:
    """Lines after the XYDATA marker, stopping at the blank line before any footer."""
    lines = path.read_text(errors="replace").splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip().upper().startswith("XYDATA"))
    body: list[str] = []
    for line in lines[start + 1 :]:
        if not line.strip():
            break
        body.append(line.strip())
    return body


@pytest.mark.parametrize(("filename", "t0"), PROBES.items())
def test_interval_scan_shape(data_dir: Path, filename: str, t0: float) -> None:
    body = _data_block(data_dir / "probe_a" / filename)
    times = [float(v) for v in body[0].split(",")[1:] if v.strip()]
    rows = [line.split(",") for line in body[1:]]

    assert len(times) == 60, "60 timepoints"
    assert len(rows) == 551, "551 wavelengths"
    assert all(len(r) == len(times) + 1 for r in rows), "wavelength column + one A per time"

    # Times are minutes -- ~10 min apart over ~10 h. In hours this would read as 25 days.
    assert times[0] == pytest.approx(t0)
    assert 570 < times[-1] < 600
    # Nominally 10 min per cuvette, but the changer jitters by up to ~0.3 min late in a
    # run (Probe f, after ~7.5 h). Use the recorded times; never assume a uniform grid.
    spacing = [b - a for a, b in pairwise(times)]
    assert all(9.7 < d < 10.3 for d in spacing), "one read per cuvette every ~10 min"

    wavelengths = [float(r[0]) for r in rows]
    assert wavelengths[0] == 800.0 and wavelengths[-1] == 250.0, "descending, needs sorting"


@pytest.mark.parametrize(
    "stem", ["1a_0h", "1a_15min", "1a_30min", "1a_1h", "1a_2h", "1a_4h", "1a_20h"]
)
def test_single_spectrum_shape(data_dir: Path, stem: str) -> None:
    path = data_dir / "1a" / f"{stem}.csv"
    body = _data_block(path)
    pairs = [(float(a), float(b)) for a, b in (line.split(",") for line in body)]

    assert len(pairs) == 801, "700-300 nm at 0.5 nm"
    assert pairs[0][0] == 700.0 and pairs[-1][0] == 300.0

    # The blank line must have cut the footer off; 'Light source,D2/WI' is not data.
    assert "Light source" in path.read_text(errors="replace")


def test_1a_saturates(data_dir: Path) -> None:
    """Kept as a negative fixture: above ~1.5 AU it is not linear in concentration."""
    peak = max(
        b
        for _, b in (map(float, ln.split(",")) for ln in _data_block(data_dir / "1a" / "1a_0h.csv"))
    )
    assert peak > 1.5


def test_buffer_blank_is_negative_in_deep_uv(data_dir: Path) -> None:
    """The deep UV is instrument baseline, not sample -- do not select spectra on it."""
    pairs = dict(
        (float(a), float(b))
        for a, b in (
            ln.split(",")
            for ln in _data_block(data_dir / "probe_a" / "20260827_Puffer_background.csv")
        )
    )
    assert pairs[250.0] < -0.15
