---
status: accepted
date: 2026-10-06
source_paper: doi:10.3982/ECTA9097
implements: [src/spectrahandler/kinetics/fit.py, src/spectrahandler/kinetics/summary.py]
---

# 0005 — Kinetic fit: maximum likelihood per replicate, uncertainty from the replicates

## Context

A kinetic model fixes the concentration profiles `C(t; k)`. That removes the rotational ambiguity
of [ADR 0003](0003-feasible-band-not-posterior.md): given the rates, the species spectra are
identified, and a point estimate with an uncertainty is meaningful again.

What that uncertainty should be was settled on a stopped-flow data set that stays local, under
the data owner's terms: four shots of one enzyme reaction from one loading, one first-order step.

1. **One joint fit claims about 8× too much precision.** It shares k and the spectra over all
   shots and treats the 400 spectra as independent. Fitted per shot, the rates spread about 8× more
   than the precision each single fit claims. The experimental unit is the shot, not the
   spectrum. A single fit over all shots pseudo-replicates.
2. **A better noise model does not fix it.** A calibrated noise covariance halves the
   overconfidence: 8× instead of 16× for independent channels. The rest is a per-shot static
   residual pattern, 3–18× the noise expectation, that frame-pair noise estimates cannot see by
   construction.
3. **The model is misspecified in a small way**, with a systematic residual at one band in every
   shot. Under misspecification the fit targets the pseudo-true parameter: the member of the
   model family closest in Kullback–Leibler divergence to the truth. Model-based intervals,
   frequentist or Bayesian, then have the wrong width (Müller 2013, Sec. 3.3; vault:
   `concepts/model-misspecification-and-pseudo-true-parameters.md`,
   `methods/sandwich-covariance-correction.md`).
4. **A shared structure across shots moves the estimate.** Sharing the spectra across shots
   (with relative baselines and delays) lowers the rate by about 3 % and makes one shot an outlier.
   Fitting each shot alone, the four shots agree within 1.0 %.
   Both agree on spread, but only the second needs no cross-shot assumptions.

## Options considered

1. **Joint fit, shared parameters, model-based interval.** This is what global/target analysis
   tools do. It is rejected by finding 1.
2. **Full Bayesian per replicate (NUTS).** The posterior width within one replicate is wrong for
   the same reasons as in finding 2. Sampling it more carefully does not change the reported
   uncertainty, and it adds runtime and convergence diagnostics to every fit.
3. **Hierarchical model across replicates.** It is the principled version of option 4, but with
   3–5 replicates the between-replicate scale is weakly identified, and its prior becomes
   decisive. It is too much machinery for a v1.
4. **Maximum likelihood per replicate, uncertainty from the spread between replicates.**

## Decision

**Option 4.**

- **Each run is fitted on its own.** The model is `D_r = C(t; k_r)·S_rᵀ + noise`: a first-order
  scheme, spectra free, and the noise sd per wavelength profiled out. The profile log likelihood
  is `−n/2·Σ_λ log RSS_λ(k)`.
- **Runs that share a condition label are replicates.** Rates are combined on the log scale:
  geometric mean, with a Student-t interval on n−1 degrees of freedom. Spectra are combined per
  wavelength on the linear scale, with the same interval.
- **The within-run curvature uncertainty is reported as a diagnostic, not as the result.** If the
  between-replicate spread is not clearly larger than it, the summary warns. With one replicate,
  the summary says that no replicate-based uncertainty exists.
- **This is still a posterior statement.** For normally distributed replicate log rates with flat
  priors on the mean and on the log of their spread, and negligible within-run error, the posterior
  of the mean log rate is exactly this t interval. It is also the replicate version of the
  sandwich correction: the sampling variance is measured from independent units instead of taken
  from the model's curvature.

## Consequences

- **Deviation from CLAUDE.md.** CLAUDE.md asks for posteriors "wherever the science allows". v1
  reports a maximum likelihood point estimate per run and a replicate interval, for the reasons
  above. The upgrade path stays open: the objective is a JAX log density, so NUTS per run, or a
  hierarchical model, can wrap the same function.
- **No replicates, no honest uncertainty.** That is stated in the result, not hidden behind a
  curvature interval.
- **The interval only covers what varies between the declared replicates.** For consecutive pushes
  from one loading, that is push-to-push variation, not preparation or day. The docs must say so.
- **Offsets are absorbed, not fitted.** In a closed scheme, a static offset and a time shift both
  land in the species spectra, and the rates do not move. This is checked to 5 decimals on that data.
  The spectra of end products are unaffected by a time shift; the spectra of the starting species
  and of every intermediate absorb it and describe the state at the first time label (corrected
  2026-10-06 after the final review: an earlier version said only the first species was affected).
- **Sequential schemes can be rate-swap ambiguous.** For A→B→C with a free intermediate spectrum,
  the swapped rates fit exactly as well (checked on synthetic data). v1 must detect and report
  this, not choose silently.
- **Misspecification is not in the interval.** It is made visible through residual diagnostics.
