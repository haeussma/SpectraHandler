---
status: accepted
date: 2026-09-20
implements: [src/spectrahandler/curve_resolution/dataset.py]
---

# 0001 — Array layout `(run, time, wavelength)` and alphabetical species order

## Context

Curve resolution factorises measured absorbance into pure component spectra `S` and
concentration profiles `C`. Three axes are in play — run, time, wavelength — plus a
species axis on the factors. Their order is a choice, and it propagates into every
likelihood evaluation, every plotting call, and later into the stoichiometric matrix of
the kinetic model.

The spec (`plans/001-bayesian-curve-resolution/spec.md` §3; removed in 3ab1596,
`git show 3ab1596~1:plans/001-bayesian-curve-resolution/spec.md`)
flags this as the one thing not simplified for v0, because kinetics and multi-run pooling
must slot in without a breaking change.

Prior art in `~/code/mcrals` uses a 2-D `SpectralSeries` — `(time, wavelength)` per run,
with multiple runs held as a dict and stacked ad hoc at resolve time. That works for
MCR-ALS, where augmentation is an explicit step, but it leaves the run axis implicit and
every multi-run operation re-derives it.

## Decision

Store everything dense, with **leading batch axes**:

```
absorbance : (n_run, n_time, n_wavelength)
time       : (n_run, n_time)
wavelength : (n_wavelength,)              shared, strictly increasing
mask       : (n_run, n_time)              True = measured
```

Three reasons, in order of weight:

1. **`vmap` over runs is free.** Leading batch dimensions are the JAX and NumPyro
   convention; a trailing run axis would need a transpose or an axis argument at every
   call site.
2. **No transposes inside the likelihood.** The per-run slice is `(n_time, n_wavelength)`,
   which is exactly `C @ S.T` for `C: (n_time, n_species)` and `S: (n_wavelength,
   n_species)`. The hot path stays free of reshapes.
3. **One code path.** A single-run experiment is `n_run = 1`, never a 2-D special case.

Separately: **`species` is sorted alphabetically and every array with a species axis
follows that order**, enforced at construction rather than by convention. The permutation
from caller order is recorded, and outputs always carry species names.

This deviates from `mcrals`, which preserves caller order. Alphabetical order is not
chemically meaningful, so `display_order` reorders at the presentation layer only;
internal arrays are never reordered.

## Consequences

- **Reversal is expensive.** Every model function, test and plotting call assumes this
  layout. Confirm before Task 0 of `plans/001-bayesian-curve-resolution/plan.md`
  is written, not after. (Plan removed in 3ab1596;
  `git show 3ab1596~1:plans/001-bayesian-curve-resolution/plan.md`.)
- A shared wavelength grid is now a hard requirement. v0 raises when grids differ;
  resampling at construction is deferred. This is a real restriction on real data — the
  fixtures in `tests/data/` use 1 nm and 0.5 nm steps and could not be combined today.
- Padding ragged runs to `n_time = max` costs memory proportional to the longest run.
  Acceptable at UV/Vis sizes (tens of timepoints); revisit if runs ever differ by orders
  of magnitude.
- Padded entries must be `NaN`, never zero — zero is a valid absorbance and would be
  silently fitted.
- Alphabetical ordering means a plot legend in memorised input order is wrong. Anything
  returning a bare array without species names is a bug.
- Interop with `mcrals`'s `SpectralSeries` needs an explicit conversion, not a cast. See
  [ADR 0002](0002-scope-boundary-against-mcrals.md).

## Status note

**Accepted 2026-09-20.** Task 0 of the plan is unblocked. Reversing this after
`dataset.py` exists is a wide refactor; supersede with a new ADR rather than editing
this one.
