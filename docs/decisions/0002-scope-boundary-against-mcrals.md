---
status: accepted
date: 2026-10-05
brain_page: ~/brain/wiki/methods/multivariate-curve-resolution.md
implements: []
---

# 0002 — Where data loading lives: SpectraHandler or mcrals

## Context

`~/code/mcrals` already does the unglamorous half of this problem, and does it against the
same instrument and the same chemistry:

- readers for JASCO exports, both interval-scan and one-file-per-spectrum
- time parsing, including the mixed-unit filename case
- composable preprocessing: trim, blank, baseline, path length, interpolation, with a
  recorded history
- QC that reports rather than raises: saturation, baseline-window contamination, mass
  balance, distinctness
- MCR-ALS resolution via pyMCR, with augmentation and closure

SpectraHandler's spec declares file readers out of scope. That is coherent for v0, which
is validated on synthetic data — but it leaves a gap with a hard edge: **nothing can turn
the real fixtures now committed in `tests/data/` into a `SpectralDataset`.** Those files
are unreachable by this library today.

The two packages also disagree structurally. `mcrals` has a 2-D `SpectralSeries` per run
with runs held in a dict; SpectraHandler has a 3-D dense array with an explicit run axis
([ADR 0001](0001-array-layout-and-canonical-ordering.md)). Neither is a cast of the other.

Worth noting what `mcrals` knows that is not in any file format: noise must be estimated
along time rather than wavelength for interpolated exports, the deep UV is instrument
baseline, 355 nm is the reductant. That knowledge cost two wrong readings to acquire. It
lives in `mcrals`'s QC checks and in `tests/data/README.md`, and it should not be
rediscovered.

## Options considered

1. Depend on `mcrals` and ship one adapter to `SpectralDataset`.
2. Absorb `mcrals`'s IO, QC and preprocessing layers here and retire it.
3. Write a minimal reader here for the two JASCO layouts and leave `mcrals` alone.

## Decision

**SpectraHandler owns data loading.** `mcrals` is deprecated: it was a quick test, not a
product, and SpectraHandler is the tool for Bayesian curve resolution. Option 1 is
therefore off the table, and options 2 and 3 collapse into one:

- Readers for the two JASCO layouts documented in `tests/data/README.md` are written here,
  in NumPy at the IO boundary, producing a `SpectralDataset` directly. No SciPy, no pyMCR.
- The QC knowledge in `tests/data/README.md` and in `mcrals`'s checks (noise estimated
  along time for interpolated exports, the deep UV as instrument baseline, saturation,
  baseline-window contamination) becomes executable here. `mcrals` is read as a reference
  for what it learned, not imported.
- `mcrals` is not a dependency, now or later, and is not kept as an MCR-ALS baseline.

Which preprocessing steps (trim, blank, baseline, binning) ship, and in what order, is a
plan-level question, not part of this decision.

## Consequences

- The QC findings in `tests/data/README.md` must end up executable somewhere. Prose in a
  README does not stop anyone estimating noise along the wrong axis.
- The choice determines whether SpectraHandler is a *library others install* — the framing
  in `CLAUDE.md` — or an inference engine behind `mcrals`. That is a positioning decision
  as much as a technical one.
- SpectraHandler is a library others install, not an inference engine behind `mcrals`.
- The dependency tree stays JAX/NumPyro only.
- The real fixtures in `tests/data/` become reachable as soon as the reader lands, which
  unblocks v1's multi-run and Probe c checks (spec §10).

## Amended 2026-10-05

- The method is the feasible band of [ADR 0003](0003-feasible-band-not-posterior.md), not
  Bayesian curve resolution: a posterior is reported only for quantities the data
  identify.
- The JASCO readers return a `Scan` of plain arrays; the user passes those to
  `SpectralDataset.create`. They do not produce a `SpectralDataset` directly.
