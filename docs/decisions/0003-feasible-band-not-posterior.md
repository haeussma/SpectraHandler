---
status: accepted
date: 2026-10-05
source_paper: doi:10.3982/ECTA16773
brain_page: ~/brain/wiki/methods/multivariate-curve-resolution.md
implements: [src/spectrahandler/curve_resolution/band.py]
---

# 0003 — Report the feasible band; a posterior only for what the data identify

## Context

Curve resolution writes a time course as `A = C·S + noise`, with `C` the concentration
profiles and `S` the pure spectra. The data fix the **product** `C·S`. They do not fix the
**split**: with the rank-K patterns `A ≈ X·Y` from an SVD, every exact split is a
re-mixing

```
C = X·T,   S = T⁻¹·Y,   T ∈ GL(K)
```

Closure (species sum to a known total) and reference spectra are linear constraints on
`T`; for three species with closure and one reference, **four** numbers of `T` stay free.
Non-negativity of `C` and `S` cuts a bounded region `Θ` out of those four dimensions.
Every point of `Θ` reproduces the data equally well. This is the rotational ambiguity of
multivariate curve resolution, and the projection of `Θ` onto a single value is its
feasible band (Lawton & Sylvestre 1971; Tauler 2001; MCR-BANDS, Jaumot & Tauler 2010;
review: Golshan et al. 2016).

**What v0 did and how it failed.** v0 put a second-order random-walk prior on each
spectrum (softplus, mean-one normalisation) and sampled with NUTS. The step-3 gate
failed: 20–43% of true concentrations inside the 95% intervals; R̂ 4.4 at the defaults;
the one converged run took 486 s and still covered 43%. Record:
`plans/001-bayesian-curve-resolution/spec.md` §7, "Gate result".

**Why it failed — the bench.** `plans/001-bayesian-curve-resolution/report-spectrum-descriptions.md`
(2026-09-22) compared 17 ways of describing `S` on synthetic data where the truth is
known, scored by coverage. Two findings decide this ADR:

1. *Every* prior placed on spectrum or concentration entries — half-normal, exponential,
   horseshoe, random walk, Gaussian process, B-spline, P-spline — covered 3–66% of the
   true amounts when the truth sat inside `Θ`, with σ recovered exactly. A prior acts in
   hundreds of dimensions; the ambiguity lives in four, and what the prior does there is
   invisible when it is written down and decisive when it is used. **Smoothness cannot
   help:** a re-mixing of smooth spectra is smooth, so a smoothness prior cannot tell the
   points of `Θ` apart. It only adds an accidental preference among them (for less total
   curvature).
2. Even a neutral prior is tilted by the likelihood's own geometry. Integrating the
   ~275 noise directions around each point of `Θ` leaves a volume factor that varies by
   13–22 nats across `Θ` and concentrates the posterior on 0.5–2% of it. In closed form
   it predicts the half-normal model's coverage as 0.18; the bench measured 0.18
   (report §A.3).

**What the theory says.** For partially identified models the posterior of the
unidentified part converges to its prior (Poirier 1998); Bayesian credible sets lie
strictly inside the frequentist confidence sets (Moon & Schorfheide 2012). Giacomini &
Kitagawa (2021) give the principled alternative: keep a posterior for the identified
part `φ` (the product and σ), admit *every* prior on the unidentified part `θ | φ`, and
report the resulting set — the robust credible region. It covers the truth at the stated
level under any prior on `θ`.

**What worked.** Describing `S` and `C` directly as re-mixings, with the band as the
interval: coverage 1.00 / 1.00 on both synthetic sets and under its own prior, no
sampler to converge (report §C.1). A JAX port written for plan 002 reproduced it: band
coverage 1.000 / 1.000 on 16 of 16 noise draws, median amount band 2.0 µM (realistic)
and 4.5 µM (harder) on a 12.5 µM total, about 1.5 s per fit.

## Decision

1. **The deliverable of curve resolution is the feasible band.** For every concentration
   and every spectrum value: the minimum and maximum over `Θ`, widened by a noise margin
   of `z_slack` propagated standard deviations, clipped at zero. Non-negativity is
   tested at `-z_slack` sd (default 3.5), so true zeros stay inside under noise.
2. **The split is parametrised explicitly** as `T`, restricted by closure and references
   to an affine subspace, with free coordinates `θ`. No prior goes on the entries of `C`
   or `S` to identify the split.
3. **A typical-solution summary is reported beside the band, and labelled as such:**
   draws spread uniformly over `Θ` (a flat prior on `θ`, stated) plus propagated noise.
   Its central interval is not claimed to be calibrated.
4. **Constraints narrow the band, priors do not.** Closure, reference spectra, runs that
   share spectra and differ in which species are present, and — later — a kinetic rate
   law enter as constraints on `T`.
5. **A posterior over the split is reported only once outside information identifies
   it**, e.g. a kinetic model that pins `T`. NumPyro is the tool for that stage and for
   the identified part `φ`.
6. **Smoothness enters on the noise part only, if at all**: denoising the spectral
   patterns before the band is computed, as a separately evaluated step. It never
   enters as a prior meant to choose between splits.
7. **v0 is retired**: the random-walk model and the NUTS `fit` driver are deleted.

### Where this deviates from the sources

- **Noise margin instead of sampling `φ`.** Giacomini & Kitagawa sample `φ` from its
  posterior and take the region holding the identified set with 95% posterior
  probability. Here the noise enters as a linearised margin: `σ‖T[:, k]‖` for amounts,
  `σ·√Σⱼ (T⁻¹[k, j] / sⱼ)²` for spectra. It is validated only empirically (16 / 16
  synthetic runs). Sampling `φ` is the upgrade path if real data disagree.
- **`z_slack` is applied twice** — as slack in the region and as the margin — so the band
  is conservative by construction. Measured coverage is 1.00, not 0.95.
- **Exploring `Θ`.** The band's extremes come from a hit-and-run walk over `Θ`, so they
  are an inner approximation of the true projection. With four free numbers and 8 000
  steps the band reproduced the bench's widths. MCR-BANDS computes the extremes by
  constrained optimisation, and polygon inflation (Sawall et al. 2013) computes `Θ`
  exactly. One of those becomes necessary as the number of free coordinates grows.
- **Two fixes over the bench's `remix.py`.** (a) `Θ` is not convex, so a uniform step can
  land in a gap; from there every bracket is empty and the chain never moves again. One
  chain key had 89% infeasible draws and flat coverage 0.25. Slice-sampling shrinkage
  (Neal 2003, doi:10.1214/aos/1056562461) redraws instead. (b) The bench started from
  rows it knew to be pure in the synthetic data; the library starts from MCR-ALS run in
  the reduced space, which also works without any reference.

## Consequences

- **The intervals are wide, and that is the result.** About 2–4.5 µM on a 12.5 µM total
  for one run with one reference. Every narrower method on the bench was 5–30× narrower
  *and wrong*. Users will ask why; the docs page says so up front.
- **Species without a reference are named arbitrarily among themselves.** The band for
  "b" can be the spectrum of "c". A reference, or a run where one of them is absent,
  pins the names.
- **Shared spectra help only when runs change which species are present.** Measured: a
  second run with no `a` narrowed the band from 4.49 to 2.99 µM; a second run with
  different rates only, from 4.49 to 4.40 µM. The Probe a/c/d design (Probe c has no
  cobalamin) is the right kind of experiment.
- **No NumPyro in `src/` until the kinetic stage.** The CLAUDE.md line "report
  posteriors wherever the science allows it" still holds — the science allows a
  posterior for identified quantities only — but the wording should say so.
- **Assumptions baked in:** the number of species is given; one white noise level
  everywhere; every run fully measured; strictly bilinear data, so a baseline or per-run
  offset shows up as an extra component.
- **Limits of validity:** this is validated for three species and four free numbers. The
  real Probe a data have six species and about twenty free numbers, where hit-and-run is
  too slow and the band is the only usable output. That needs the AFS or optimisation
  route above, and it gets its own plan.
