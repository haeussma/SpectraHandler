---
status: accepted
date: 2026-10-07
implements: [src/spectrahandler/plot.py]
---

# 0006 — Plotting API: one run per panel, on the caller's axes

## Context

Until now `spectrahandler.plot` held two figure functions with fixed layouts. `plot_kinetic_fit`
drew one three-panel figure over all runs and averaged the species spectra across runs.

Real use needed more than that. The first real analysis (four replicate shots of one
experiment) produced seven figures for the analysis and seven more for slides. They shared the same
panels but needed different layouts, fonts and species names. Each panel was rewritten by hand in
every script.

Three things came out of that work:
- Max wants every shot on its own axes, compared side by side, and **no averages**: a mean
  spectrum hides exactly the disagreement between shots one wants to see.
- **Deconvolution matters with or without a kinetic model.** `FeasibleBand` is a deconvolution
  without one.
- **Scientific plots carry no titles.**

## Decision

1. **One module.** All plotting stays in `spectrahandler.plot`, the only module that imports
   matplotlib (optional `plot` extra). Domain packages stay free of it.
2. **Panel functions, one run each, by run id.** Every `plot_*` function draws one run,
   addressed by its id from `run_ids` and never by index. It draws on axes the caller passes,
   or on a new figure, and returns them. Layout, figure size and fonts belong to the caller,
   through rcParams and `plt.style.context`.
3. **A deconvolution is drawn whole.** `plot_deconvolution` draws concentrations and species
   spectra together, on two axes, for a `KineticFit` (point estimate plus kinetic model) or a
   `FeasibleBand` (shaded band). The kinetic model is a layer, not the subject.
4. **Residuals only where there is one residual.** `plot_residuals` takes a `KineticFit`; a band
   has none.
5. **Calculations live in the core, not in plotting.** The projection of the data onto species
   spectra is `SpectralDataset.project`. The replicate posterior is `RateEstimate.density`.
   Plot functions only draw.
6. **Every result has a run axis.** `FeasibleBand` spectra carry a run axis of size 1 (one set
   shared by all runs), so per-run and shared spectra broadcast alike.
7. **No titles, no single-wavelength traces, no averages across runs.** The run id is written
   inside the axes. Species colours are a fixed public palette, never cycled. `colors=` and
   `labels=` are keyed by species name.

## Consequences

- **Grids are user code**, a short loop over `run_ids`. There is no figure-level function for
  it until one is clearly needed.
- **A third result type** (an unconstrained MCR fit, say) will need `plot_deconvolution` to accept
  it. That is the moment to replace `KineticFit | FeasibleBand` with a protocol over `spectra` and
  `concentrations`.
- **A non-time second axis** (titration, temperature) plots with "time" labels until
  `SpectralDataset` gains a general axis name.
- **Removing `plot_kinetic_fit` is a breaking change** before 1.0. The guides show the
  replacement.
