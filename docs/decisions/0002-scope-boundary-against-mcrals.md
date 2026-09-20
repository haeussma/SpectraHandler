---
status: proposed
date: 2026-09-20
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

## Decision

**Not yet made.** Three candidates:

1. **Depend on `mcrals`.** SpectraHandler takes it as a dependency and ships one adapter,
   `SpectralSeries` → `SpectralDataset`. Cheapest; no duplicated readers. Costs: an
   unpublished git dependency, `mcrals` pulls in pyMCR and SciPy, and its API is not
   stable.
2. **Absorb the IO, QC and preprocessing layers** into SpectraHandler and retire `mcrals`
   to its MCR-ALS baseline role. One package for users. Costs: a real port, and the
   inference work stalls behind it.
3. **Write a minimal reader here** for only the two JASCO layouts documented in
   `tests/data/README.md`, and leave `mcrals` alone. Smallest immediate diff. Costs: two
   packages parsing the same files, and the QC knowledge duplicated or lost.

The decision does not block steps 0–3 of the plan, which use synthetic data only. It
blocks step 8 and anything published.

## Consequences

Whichever is chosen:

- The QC findings in `tests/data/README.md` must end up executable somewhere. Prose in a
  README does not stop anyone estimating noise along the wrong axis.
- The choice determines whether SpectraHandler is a *library others install* — the framing
  in `CLAUDE.md` — or an inference engine behind `mcrals`. That is a positioning decision
  as much as a technical one.
- Option 1 makes SpectraHandler's dependency tree include SciPy and pyMCR, which sits
  awkwardly with the JAX-only rule in `CLAUDE.md`.

Revisit before step 8, and before the first release.
