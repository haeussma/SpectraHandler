---
status: accepted
date: 2026-11-19
source_paper: doi:10.3982/ECTA16773
implements: [src/spectrahandler/curve_resolution/band.py]
---

# 0007 — Credible band: a posterior for the identified part, a set for the split

## Context

[ADR 0003](0003-feasible-band-not-posterior.md) decided that curve resolution reports the
feasible band and that a posterior is given only for what the data identify. It left one
deviation from its source open: the noise entered as a linearised margin around one SVD,
with one noise level everywhere, and "sampling `φ` is the upgrade path". This ADR takes
that path. The goal is a band with a stated credibility level, a proper noise model, and
the noise posterior as a result, still without a kinetic model.

**The framework.** Giacomini & Kitagawa (2021) split the parameters into an identified
part `φ` and the rest. `φ` gets an ordinary posterior. The rest gets no single prior: for
every `φ` the whole identified set is kept, and the robust credible region is the set
that contains it with the stated posterior probability. Here `φ` is the rank-K product
`C·S` and the noise; the rest is the re-mixing `T`.

**What the literature does** (see [`../literature.md`](../literature.md)). Bayesian
unmixing and Bayesian NMF put one prior on spectra and abundances and report its
posterior. Chemometric feasible-band methods for noisy data (N-BANDS, FACPACK) keep the
set but give it no probability level: they take one SVD as exact, assume one noise level,
and tolerate small negative values up to a threshold that is partly set by hand. No paper
found combines the two.

**Two spikes decided the design** (throwaway scripts, 2026-11-19; synthetic a → b → c with
one reference, 29 spectra × 64 wavelengths, σ = 0.002 AU; "realistic" has true zeros in
the spectra, "harder" lifts every spectrum off zero).

1. *Sample non-negative `C` and `S` with NUTS; for each draw take every exact
   non-negative re-mixing; report the 95 % envelope.* This follows Giacomini & Kitagawa
   literally. On the harder set it covered 0.99–1.00 of the true amounts and 0.95–0.98 of
   the true spectrum values. On the realistic set it covered 0.93–0.98 and only 0.72–0.76
   (0.85–0.93 with a one-sided interval at zero); in one run 22 of 64 channels of the
   intermediate's spectrum lay outside. The cause: where a true spectrum is zero, each
   draw holds small positive noise, and exact non-negativity of that draw stops the set
   short of the truth. A truth with many exact zeros sits on a corner that a continuous
   posterior never reaches. The unidentified directions also mixed badly (R̂ up to 1.55
   for concentrations, up to 29 divergences).
   The same runs reproduced ADR 0003: the plain posterior of that model covered 0.12–0.17
   of the true amounts on the harder set.
2. *Sample the identified part without constraints in a fixed gauge; test non-negativity
   against its posterior.* 13 runs over both sets, six noise seeds, and noise that grows
   threefold with wavelength: 1.00 of the true amounts in every run, 0.96–0.99 of the true
   spectrum values, R̂ ≤ 1.02, no divergences, about 5 s per fit.

## Decision

1. **Stage 1, the identified part, is a NumPyro model.**
   `A[t, λ] ~ Normal((X·Y)[t, λ], σ_λ)`, independent between all points, one `σ_λ` per
   wavelength. Each reference spectrum is one more row, `Normal(x_ref·Y, σ_ref)`.
2. **The gauge is fixed**, so the posterior has one mode:
   `Y = (y₀ + B·y_⊥)·σ⁰`, with `y₀` the leading K right singular vectors of the
   noise-scaled data, `y_⊥` their complement, and `B` free. Flat priors on `X`, `B` and
   `x_ref`; a half-normal prior on `σ_λ` whose scale defaults to the RMS absorbance.
   Sampled with NUTS, two chains by default; a split R-hat above 1.05 raises a warning.
3. **Stage 2, the split, gets no prior.** `C = X·T`, `S = T⁻¹·Y`. Closure and references
   are equalities on `T`, evaluated at the posterior mean. For a given `T`, every
   concentration and spectrum value is a linear function of one row of `X` or one column
   of `Y`, so its posterior mean and standard deviation follow from the stage-1 draws.
4. **A re-mixing is feasible unless some value is below zero by more than `z_region`
   posterior standard deviations.** `z_region` is the normal quantile at
   `level^(1/n_value)`: the true split passes all `n_value` tests together with
   probability `level`. For 282 values and 95 % it is 3.56.
5. **The band is the range, over the feasible region, of each value's central `level`
   posterior interval** (`mean ± 1.96 sd` at 95 %), clipped at zero.
6. **Outputs:** the band; its ambiguity part (extremes of the posterior mean); flat draws
   over the region as a labelled typical-solution summary; the posterior of `σ_λ`; the
   level; R-hat.
7. **`sigma` can be given** as one number or one per wavelength; it is then fixed and not
   sampled.
8. **No backward compatibility.** `FeasibleBand` is replaced by `CredibleBand`, and
   `z_slack` by `level`.

This amends ADR 0003 in two places: NumPyro is back in `src/` before the kinetic stage,
and the linearised noise margin is gone. Its core stands: no prior identifies the split.

### Where this deviates from the sources

- **Non-negativity is a posterior test, not exact per draw.** Giacomini & Kitagawa take
  the exact identified set of every draw of `φ`. Spike 1 shows that this under-covers a
  truth with exact zeros. Testing each value against its posterior standard deviation is
  the tolerance that the chemometric methods apply by hand, derived here from the level.
- **The region is conservative, the margin is pointwise.** The joint test keeps the true
  split with probability `level`; the interval then covers each value with probability
  `level` given the split. The stated level is therefore approximate. Measured coverage is
  0.96–1.00, and near 0.95 only for a species pinned by a reference, where the band is an
  ordinary posterior interval.
- **Normal approximation.** Means and standard deviations of the draws stand in for the
  posterior of each value. The stage-1 posterior is close to normal by construction.
- **Equalities at the posterior mean.** The uncertainty of closure and of the reference
  rows enters through the standard deviations, not through the equalities.
- **Exploring the region** is unchanged from ADR 0003: hit-and-run with slice shrinkage
  (Neal 2003), an inner approximation.

## Consequences

- **Assumptions:** independent normal noise with one level per wavelength; the number of
  species is given; every run fully measured; strictly bilinear data. A baseline or
  background is still an extra component; [ADR 0004](0004-backgrounds-need-a-zero-window.md)
  says how it will be added.
- **Independence is assumed, not checked by the fit.** Data smoothed or interpolated along
  wavelength break it and make the inferred noise too small. `noise_diagnostics` shows the
  residual autocorrelation; the remedy is to pass `sigma`.
- **With few spectra the noise posterior is wide.** Each `σ_λ` rests on `n_spectra − K`
  residuals: with 30 spectra single values scatter by about ±40 % around the truth.
- **The band's width is about that of ADR 0003** (median 1.9–2.1 µM on the realistic set,
  4.4–4.5 µM on the harder set, of 12.5 µM), because `z_region` ≈ the old `z_slack`. What
  changed is that the level is stated, the noise is per wavelength and inferred with its
  uncertainty, and the uncertainty of the SVD subspace is included.
- **Validated for three species and four to six free numbers, on synthetic data only.**
  The limit of ADR 0003 stands: for six species hit-and-run is too slow. The column-wise
  sampler of Sawall et al. (2025) is the candidate replacement.
- **The stage-1 model is the place to add a kinetic model later**: replacing the free `X`
  by `C(t; k)` pins `T`, and the same noise model carries over.
