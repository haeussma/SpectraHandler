# Describing pure spectra inside a curve-resolution model — which description gives credible uncertainty?

**Status:** evidence report · **Date:** 2026-09-22 · **Branch:** `feat/bayesian-curve-resolution` ·
**Scope:** synthetic data only; throwaway scripts; nothing under `src/` or `tests/` changed.

**Question.** `A(t, λ) = Σ_k c_k(t) s_k(λ) + noise` pins down the product `C·S` but not the split
into `C` and `S`. With the known total (12.5 µM) and one substrate scan (50 µM) the split has
**four** free numbers. Which way of describing the pure spectra `s_k(λ)` gives 95% intervals
that contain the truth about 95% of the time?

**Criterion.** Coverage, not fit quality: over 16 noise draws per dataset, the fraction of
true amounts (29 × 3) and true spectrum values (3 × 64) inside the central 95% interval, with
the posterior σ required to match the true 0.002 and R-hat computed after relabelling every
chain onto the truth by spectral correlation.

**Answer.** Describe the pure spectra as re-mixings of the data's own rank-3 patterns with an
explicit, flat prior on the four free re-mixing numbers, and report the feasible band (± noise)
as the interval. It was the only description that contained the truth at least 95% of the time
on both datasets and under its own prior, at 13 s per fit and with no sampler. Every entry-wise
or smoothness prior on the spectrum values — half-normal, exponential, horseshoe, second-order
random walk, Gaussian process through a softplus or a log link, B-spline, P-spline — covered
3–66% of the amounts on the interior-truth data, with σ exactly right, R-hat mostly above 1.05,
and intervals a fifth to a third of the band. The cause is only partly the prior: the
likelihood's own volume factor, derived in closed form in A.3, concentrates the split onto
0.5–2% of the feasible region and predicts the half-normal model's coverage (0.18 predicted,
0.21 measured); adding its correction lifts the half-normal to 0.81 and a hard kinetic model to
0.94. Parametric band models identify the split and are calibrated when the family is right
(0.94 / 0.93) and confidently wrong when it is not (0.05). Smoothing the data before pattern
extraction narrows the spectral noise margin by the filter's own factor, leaves the ambiguity
untouched, and costs coverage on narrow bands.

## 0. The evidence in one table

Coverage of the central 95% interval, amounts / spectra, mean over 16 noise draws (8 in prior
mode); "harder" is the set with no exact zeros, i.e. the clean test of the split. Full tables
with per-species coverage, widths, R-hat, divergences, σ and wall clock in B.2.

| description of `S` | harder, truth | realistic, truth | own prior | width of amounts, harder (µM) | verdict |
| --- | --- | --- | --- | --- | --- |
| **re-mixing band** (iv) | **1.00 / 1.00** | **1.00 / 1.00** | **1.00 / 1.00** | 4.5 | calibrated by construction; the deliverable |
| re-mixing, flat posterior (iv) | 0.99 / 0.97 | 0.93 / 0.83 | 0.97 / 0.96 | 3.2 | calibrated on interior truths; under-covers a boundary truth, as theory says |
| re-mixing, mild zero preference (iv) | 0.99 / 0.96 | 1.00 / 0.95 | 0.90 / 0.91 | 3.4 | helps where spectra have zeros; costs 5% under its own prior |
| Gaussian bands in nm, counts known (iii) | 0.30 / 0.05 | 0.88 / 0.84 | 0.96 / 0.98 | 1.0 | identifies; right family → 0.94 / 0.93; offset in the data → σ 7× off |
| bands in nm + offset (iii) | 0.94 / 0.92 | 0.94 / 0.65 | 0.96 / 0.92 | 0.15 | calibrated and 30× narrower when the family contains the truth |
| bands in wavenumber (iii) | 0.16 / 0.07 | 0.40 / 0.06 | 0.91 / 0.93 | 1.0 | misspecified by 8% skew: confidently wrong |
| half-normal on entries | 0.21 / 0.68 | 0.74 / 0.57 | 0.99 / 0.97 | 1.1 | neutral prior, tilted by the likelihood's geometry |
| half-normal + volume correction | 0.81 / 0.91 | 0.61 / 0.52 | 0.97 / 0.95 † | 2.6 | the geometry term removed: most of the way back |
| exponential on entries (v) | 0.34 / 0.64 | 0.76 / 0.56 | 0.96 / 0.96 | 1.3 | tilt toward minimum total absorbance |
| horseshoe (v) | 0.04 / 0.51 | 0.42 / 0.63 | 0.64 / 0.83 | 0.3 | prefers zeros hard; sampler fails its own prior |
| RW2 on channels, softplus (ii; v0) | 0.16 / 0.68 | 0.46 / 0.46 | 0.37 / 0.68 | 0.3 | 10³–10⁴ nats of opinion; never converges |
| GP Matérn-3/2, softplus (ii) | 0.03 / 0.51 | 0.51 / 0.70 | 0.92 / 0.95 | 0.3 | mildest smoothness tilt, still fails on the split |
| GP Matérn-3/2, log link (ii) | 0.04 / 0.51 | 0.77 / 0.69 | 0.83 / 0.90 | 0.4 | excludes the boundary of Θ |
| B-spline, exponential weights (i) | 0.66 / 0.85 | 0.73 / 0.49 | 0.94 / 0.95 | 1.6 | inner approximation of Θ plus a linear tilt |
| P-spline (i) | 0.08 / 0.50 | 0.29 / 0.55 | 0.36 / 0.63 | 0.3 | quadratic tilt; does not converge |
| hard kinetic model on C, free spectra | 0.73 / 0.91 | 0.15 / 0.68 | 0.92 / 0.95 | 0.15 | identified, yet biased by the same geometry and by the boundary |
| kinetic + volume correction | 0.94 / 0.94 | 0.14 / 0.67 | 0.75 / 0.88 † | 0.14 | calibrated on the interior truth |

† The volume factor enters as a `numpyro.factor`, which changes the effective prior, while the
prior-mode truth is drawn from the uncorrected generative prior: for these two rows the check is
mismatched by construction and the number is not a calibration failure. The corrected kinetic
row's own-prior draws include rate pairs (LogNormal, 0.07–3.7 h⁻¹) for which the reaction is over
before the second scan, where the profile likelihood is far from normal.

---

## A. Where each description puts its prior mass in the re-mixing directions

### A.1 The identification structure

Stack the time course (29 scans, first scan dropped) and the substrate scan into one matrix
`D` (30 × 64). A rank-3 SVD gives patterns `D ≈ X·Y` with `X` (30 × 3) and `Y` (3 × 64). Every
exact factorisation is a re-mixing of that one:

```
C = X·T,   S = T⁻¹·Y,   T ∈ GL(3)          (9 numbers)
```

The known facts are linear in `T`: every time row of `C` sums to 12.5 and the scan row to 50
(`X·T·1 = d`, so `T·1 = t*`: 3 equations), and the scan row has composition (50, 0, 0)
(`x_refᵀ·T = (50, 0, 0)`: 3 equations). One of the six is implied by the others (the scan's
row sum), so the equalities have rank 5 and **`n_free = 9 − 5 = 4`** (measured; 2 with a
second reference scan). Non-negativity, `X·T ≥ 0` and `T⁻¹·Y ≥ 0`, then cuts a bounded region
`Θ` out of that 4-dimensional affine space. Every point of `Θ` reproduces `D` exactly; the data
cannot rank them. The projection of `Θ` onto one value, say `c_b(t₅)`, is its **feasible band**.

This is the field's founding result and its 50-year elaboration: Lawton & Sylvestre (1971) for
two components; general bands by constrained optimisation (Gemperline 1999; Tauler 2001;
MCR-BANDS, Jaumot & Tauler 2010; N-BANDS with noise, Olivieri & Tauler 2021); the region
itself, the *area of feasible solutions*, computed analytically (Rajkó & István 2005) or by
polygon inflation (Sawall et al. 2013), reviewed in Golshan et al. (2016); the conditions
under which the band collapses to a point — selective windows, zero regions, known spectra —
are Manne's resolution theorems (Manne 1995) and Abdollahi & Tauler (2011); de Juan & Tauler
(2021) is the review of record, and Olivieri (2025) is specifically about first-order
(UV/Vis-type) spectra, where the ambiguity is at its worst because nothing is selective.

Two consequences frame everything below. First, `Θ` is a set, not a point, and its size is a
property of the *data*; no prior can shrink it, only change how mass is spread over it.
Second, `Θ` is small in dimension (4) but the objects people put priors on are large
(87 amounts, 192 spectrum values). The prior lives in 279 dimensions; only 4 of them are
unidentified; the rest are noise directions the likelihood controls. What a prior does in
those 4 directions is invisible when you write it down and decisive when you use it.

### A.2 What the partial-identification literature says

Write the model in the transparent parametrisation of Gustafson (2005, 2015): an identified
part `φ = (X, Y, σ)` — the product and the noise — and a non-identified part `θ ∈ Θ`, the
four re-mixing numbers. Then:

- **The posterior of `θ` converges to the prior conditional.** Poirier (1998): as data
  accumulate, `p(θ | D) → p(θ | φ̂)`. The data update `θ` only through whatever dependence
  the prior built between `θ` and `φ`. Gustafson (2005) makes the same point for
  mismeasured-variable models: posterior width on the non-identified part does not shrink
  with `n`, and sensitivity to the prior does not vanish.
- **Credible ≠ confidence.** Moon & Schorfheide (2012) prove that for set-identified
  parameters Bayesian credible sets are, asymptotically, strictly *inside* the frequentist
  confidence sets. A 95% credible interval on `c_b(t)` covers the truth 95% of the time only
  if the prior on `θ | φ` happens to be right — and "right" here means: the truth sits where
  the prior puts its mass, which nobody can know from the data.
- **The fix is multiple priors, not a cleverer prior.** Giacomini & Kitagawa (2021): keep one
  prior on `φ`, admit *every* prior on `θ | φ`, and report the posterior lower and upper
  probabilities. The resulting "robust credible region" contains the identified set and,
  asymptotically, matches the frequentist confidence set for it. Translated: **the classical
  feasible band, plus a noise margin, is the robust-Bayes answer**, and any single-prior
  posterior narrower than the band is reporting its prior on the split (principle 6 of the
  spec, now with a theorem behind it). Manski (2003) is the general framing.

### A.3 The induced posterior over the split has two sources of opinion, and the second is larger

For any prior `p(C)·p(S)` and Gaussian noise, in the small-noise limit the marginal posterior
over the four numbers is

```
p(θ | D)  ∝  p_C(X·T_θ) · p_S(T_θ⁻¹·Y)  ·  V(θ)
```

The first factor is the prior's density *along the manifold* of exact solutions — what
everyone thinks about. `V(θ)` is what nobody thinks about: the integral of the likelihood
over the ~275 *noise* directions around the point `(X·T_θ, T_θ⁻¹·Y)`. By Laplace,
`V(θ) ≈ pdet(J_θᵀ J_θ)^(−1/2)`, with `J_θ` the Jacobian of the bilinear map in the model's
free coordinates, whose null space is exactly the four re-mixing directions.

For this problem `J_θᵀ J_θ` has closed-form blocks — `I ⊗ G` for the amounts under closure
(`G = S_d S_dᵀ`, `S_d` the two spectrum differences against species c), `M ⊗ I` for the
spectra (`M = CᵀC + rᵀr`, `r` the scan composition), and `C ⊗ S_d` between them — and its
pseudo-determinant reduces to

```
log pdet(JᵀJ) = n_w · log det M  +  (n_t − 2) · log det G  +  2 · log(r M⁻¹ rᵀ)    (+ const)
```

(verified against the dense 250 × 250 eigen-decomposition: agreement to 10⁻⁴ nats across the region). Under a re-mixing
`det M` scales like `|det T|²` and `det G` like `|det T|⁻²`, so `V(θ) ∝ |det T|^(n_t − n_w)`
to leading order: each spectrum contributes `n_w` noise directions whose posterior width is
`σ/‖c_k‖`, each amount contributes `n_t` directions of width `σ/‖s_k‖`, and with 64 channels
against 29 timepoints the two do not cancel. This is a Neyman–Scott-type effect — hundreds
of nuisance directions marginalised at once — and it means **even a perfectly neutral prior
on `S` and `C` gives a biased split.** Measured over `Θ` by hit-and-run (script `tilt.py`),
the 5–95% spread of `log V` is 12.9 nats on the realistic set and 22.1 nats on the harder
set: the likelihood's own geometry concentrates the split onto 2% and 0.5% of the feasible
region. On the harder set it puts the truth 17.8 nats below its preferred re-mixing, and it
**predicts the half-normal model's amount coverage as 0.18 (a/b/c: 0.14/0.31/0.10); the
bench measures 0.18 (0.05/0.41/0.09)** — the mechanism is confirmed to two digits.

The geometry is fixed by the data layout, so no choice of `p(S)` repairs it. Two things do:
parametrise the split explicitly, so that the noise directions are not parameters and `V` is
constant by construction (description iv); or add `+½·log pdet(JᵀJ)` to the model as a
`numpyro.factor`, which cancels `V` exactly and is tested in B as `halfnormal_vol`.

### A.4 Description by description

The prior term `log p_S(T⁻¹Y)` as a function of the re-mixing, for each candidate, with the
tilt measured over `Θ` (5–95% spread in nats; "effective volume" = the share of `Θ` that
carries the induced prior mass; "truth rank" = the fraction of `Θ` the prior prefers over the
truth). Evaluations are at the denoised re-mixing (smoothed, floored at the noise level),
because a sampler is free to move within the noise; without that, every link function
returns −∞ at a noisy zero.

**(i) B-spline and P-spline bases with non-negative weights.** `s_k = B·w_k`, `w_k ≥ 0`. The
spline space is closed under re-mixing (`T⁻¹Y` is again a spline when `Y` is), but `w ≥ 0`
is strictly stronger than `s ≥ 0`: the description reaches only an inner approximation of
`Θ` (90–100% of it here). With independent Exponential weights, `log p = −1ᵀ(T⁻¹W)1 / w̄` is
*linear* in `T⁻¹` — an exponential tilt toward the vertex of `Θ` with the smallest total
weight, of strength (total weight)/`w̄` ≈ `n_b · K` nats: measured 5.6 / 10.7 nats, effective
volume 0.20 / 0.011. The P-spline (Eilers & Marx 1996: the same basis with a second-difference
penalty on the coefficients) turns that into a quadratic form of the roughness Gram matrix,
`−½ tr(T⁻¹ (W D₂ᵀ D₂ Wᵀ) T⁻ᵀ)/τ²`: it prefers the re-mixing with the smallest *total*
roughness. A rotation of smooth spectra is still smooth — but not equally smooth, so the
penalty is not neutral, and its strength grows as `1/τ²`: measured 22 / 49 nats, effective
volume 0.09 / 0.11.

**(ii) Gaussian processes.** `log p = −½ Σ_k s_kᵀ K⁻¹ s_k = −½ tr(T⁻¹ Y K⁻¹ Yᵀ T⁻ᵀ)`: the
GP energy of the re-mixed patterns, again quadratic in `T⁻¹`. The second-order random walk is
the state-space form of the integrated-Wiener GP and the smoothing spline is its posterior
mean (Kimeldorf & Wahba 1970), so RW2, P-spline and GP differ in cost and in the *strength* of
this tilt, not in its direction. A Matérn-3/2 with ℓ = 20 nm and amplitude 3 through a
softplus link is the mildest of the smoothness priors here (4.3 / 6.5 nats, effective volume
0.47 / 0.66) because every re-mixing of these patterns is "smooth enough" at 20 nm. With a
**log link** the tilt doubles (9.6 / 14.4 nats) for a specific reason: `log s → −∞` where a
spectrum is zero, so a log-link GP puts *no* mass on the boundary of `Θ` — and the boundary
is where real, baseline-corrected spectra live (21% of the realistic truth's values are
exact zeros). The RW2 as used in v0 (τ = 0.05 per channel, softplus link) is the extreme
case: 1 100 / 4 400 nats spread, effective volume 4·10⁻⁴; on the realistic set the truth is
960 nats below the prior's favourite re-mixing, because a walk with that τ cannot descend
into the foot of a band. That is finding 1 in numbers. A learnable ℓ does not remove the
tilt, it lets the split pull ℓ (the bench's `gp_learn` ran to ℓ = 157 nm in the pilot).

**(iii) Parametric band models (sums of Gaussians).** `s_k(λ) = Σ_j h_kj·g(λ; μ_kj, w_kj)`.
A re-mixing `s_b + ε s_a + δ s_c` of band-sum spectra has `n_b + n_a + n_c` bands, so with
the band counts fixed the prior puts **zero** mass on every re-mixing except the identity, up
to what the widths can absorb. This description *identifies* rather than regularises — the
same structural move as Manne's selectivity or a hard kinetic model, made on the spectra —
and it is the only nonparametric-free way to get a narrow *and* honest interval. The price
is that it is either exactly right or confidently wrong: a Gaussian in wavenumber is not a
Gaussian in wavelength (the skew is `2w/λ` ≈ 8% at 355 nm), a constant 0.003 AU/µM offset is
not a band, and at σ = 0.002 with 1 920 points the likelihood detects both and converts them
into bias with narrow intervals; the posterior over band positions is multimodal (32-start
MAP needed before NUTS); and with more bands than the chemistry has, the spare bands recreate
part of the ambiguity.

**(iv) Low-rank patterns from the data with an explicit re-mixing (the "re-mixing method").**
`S = T⁻¹Y`, `C = X·T`, `T` restricted to `Θ` with a flat prior on the four free numbers,
measurement noise propagated per draw. The prior on the split is flat *by construction and
visible*; `V(θ)` is constant because the noise directions are not parameters; the band
(`Θ` projected, ± margin) is Giacomini & Kitagawa's robust credible region. What it gives up:
it does not smooth (the patterns carry the noise of a rank-3 projection, ~σ/√s_k per
channel); `Θ` must be computed with a slack for the noise (3.5σ per entry, simultaneous over
~280 entries — at 2σ the truth was outside `Θ` for 5 of 8 draws); it needs `K` known; and a
4-dimensional hit-and-run is easy while the `K² − 5` dimensions of five components are not.

**(v) Sparse and zero-preferring priors.** Exponential on the entries: `−1ᵀ T⁻¹ Y 1 / λ̄`, a
linear tilt toward the re-mixing with the smallest *total* absorbance (6.6 / 16.5 nats,
effective volume 0.32 / 0.017). The horseshoe (Carvalho, Polson & Scott 2010),
`Σ log log(1 + 2τ²/s²)`, prefers exact zeros hard: on the realistic set it ranks the truth in
the top 0.1% of `Θ` — for the wrong reason (it likes zeros; the truth has zeros) — while
still concentrating on 0.4% of `Θ`; on the harder set, where nothing is zero, it concentrates
on 0.3% of `Θ` 14 nats away from the truth. A Dirichlet on the spline *shape* with α < 1 is a
logarithmic tilt `(α − 1) Σ log w`, i.e. an infinite preference for exact zeros of the
weights. The mild "zero preference" of finding 2 (β = 0.1–0.3 per practically-zero value,
applied on `θ`) is the same idea in the one place where its strength can be read off: a few
nats in total, and it is the only one of these that does not break — precisely because it is
weak and explicit.

### A.5 Measured tilt over the feasible region

| description | spread of log prior over Θ, 5–95% (nats) real / harder | effective volume of Θ real / harder | truth rank real / harder | truth deficit (nats) real / harder |
|---|---|---|---|---|
| HalfNormal(0.05) on entries | 0.2 / 0.9 | 0.998 / 0.943 | 0.773 / 0.026 | 0.2 / 0.1 |
| Exponential, mean 0.01 | 4.1 / 8.8 | 0.318 / 0.017 | 0.048 / 0.028 | 1.1 / 3.7 |
| horseshoe, global scale 0.01 | 16.0 / 14.0 | 0.004 / 0.003 | 0.001 / 0.156 | 0.9 / 14.1 |
| B-spline, Exponential weights | 5.6 / 10.7 | 0.204 / 0.011 | 0.049 / 0.040 | 1.4 / 4.7 |
| P-spline (RW2 on linked weights, τ = 0.5) | 22.4 / 48.5 | 0.088 / 0.108 | 0.598 / 0.008 | 10.4 / 0.4 |
| GP Matérn-3/2, ℓ = 20 nm, softplus link | 4.3 / 6.5 | 0.472 / 0.657 | 0.973 / 0.075 | 5.2 / 0.2 |
| GP Matérn-3/2, ℓ = 20 nm, log link | 9.6 / 14.4 | 0.208 / 0.522 | 0.973 / 0.101 | 11.5 / 0.4 |
| RW2 per channel, τ = 0.05, softplus link (v0) | 1067.6 / 4394.1 | 0.000 / 0.004 | 0.913 / 0.004 | 959.9 / 2.3 |
| likelihood volume factor V(θ) alone (flat prior) | 12.9 / 22.1 | 0.019 / 0.005 | 0.019 / 0.370 | 2.4 / 17.8 |

Feasible region: 4 free numbers; 2 750 flat draws per dataset (hit-and-run, 3.5σ simultaneous slack); the truth is inside Θ on both datasets. Reachable fraction of Θ for the spline descriptions (weights ≥ 0 within noise): 1.00 / 0.90. The manifold's own volume element is constant over Θ to < 0.01 nats in these units, so "flat in the four numbers" and "flat on the manifold" coincide here.

Predicted amount coverage of the split posterior (prior tilt × V, region only, no noise margin) versus the bench, harder set: half-normal 0.18 predicted (a/b/c 0.14/0.31/0.10) — see B.2 for the measured value.


The table has one lesson. Every entry-wise description either concentrates the split onto a
few percent of `Θ` (RW2, P-spline, horseshoe, exponential, B-spline) or is nearly neutral
(half-normal; GP through softplus at 20 nm) — **and the nearly neutral ones fail anyway**,
because `V(θ)` concentrates the split for them. Neutrality of the prior is necessary and not
sufficient; the geometry must be handled as well.

---

## B. Test bench

### B.1 Design

Two synthetic truths from `make_realistic_dataset` (a → b → c, k₁ = 0.6 h⁻¹, k₂ = 0.4 h⁻¹,
30 scans over 10 h, 64 channels 340–700 nm; two-band a and c, one-band b; absorbance ≤ 0.33):

| dataset | spectra | exact zeros in the truth |
| --- | --- | --- |
| realistic | as generated | 21% of spectrum values < 10⁻⁶ AU/µM (a/b/c: 11/44/8%); 3.4% of amounts < 0.05 µM |
| harder | + 0.003 AU/µM constant on every spectrum | none |

Always: first scan dropped (29 time rows), one substrate scan at 50 µM with the same noise,
total 12.5 µM known and used as closure, σ = 0.002 added fresh per seed (`jax.random.key(1000
+ seed)`, 16 seeds). Every NUTS approach shares the same amounts model — `12.5 ·
Dirichlet(1, 1, 1)` per scan — the same likelihood and the same σ ~ HalfNormal(0.01); only
the spectrum description changes. Fits: 2 chains, 1 500 warmup + 1 500 draws, target accept
0.9, both chains started at the best of 32 prior-drawn multi-start MAP optima (except the
horseshoe, whose joint mode is degenerate, and the plain `kinetic` row, kept at default
inits so that R-hat can show its two modes). Scoring: per-chain relabelling onto the truth
by cosine similarity of the posterior-mean spectra, then pooled central 95% intervals; split
R-hat and ESS over the relabelled spectra, amounts and σ; divergences; posterior σ; wall clock
after `jax.block_until_ready`. "Prior mode" draws `(S, C, σ)` from the approach's own prior
(8 draws), simulates, fits — the check that the code and the sampler are right, which
should give ≈ 95% for every approach regardless of its merits on real truths.

The re-mixing method (`remix.py`) is not sampled with NUTS: rank-3 SVD of the stacked data,
the equality facts solved for `T` up to 4 free numbers, `Θ` found by random descent and
explored by hit-and-run (8 000 steps), each draw perturbed by its propagated noise; three
interval kinds are reported — `remix_flat` (central 95% of flat-over-`Θ` + noise),
`remix_zero` (the same with draws re-weighted by `exp(0.3 · #practically-zero values)`), and
`remix_band` (min/max over `Θ` ± 2σ, the safety check) — with and without smoothing the data
along wavelength before the SVD (local quadratic, 5 or 7 channels).

Approaches: `halfnormal` (independent HalfNormal(0.05) — the finding-1 baseline);
`halfnormal_vol` (the same plus the `+½ log pdet(JᵀJ)` factor of A.3); `exponential`
(mean 0.01); `horseshoe`; `rw2` (v0's walk, τ = 0.05, softplus, closure-scaled) and
`rw2_loose` (τ = 0.2); `gp_softplus` and `gp_log` (Matérn-3/2, ℓ = 20 nm); `gp_learn`
(ℓ ~ LogNormal(log 20, 0.35)); `bspline_exp` (cubic, 10 nm knots, Exponential weights);
`pspline` (same basis, RW2 on the softplus-linked weights); `bands_nm` (Gaussians in nm,
counts 2/1/2), `bands_wn` (Gaussians in wavenumber, 2/1/2), `bands_free3` (3 bands each, nm),
`bands_nm_off` (2/1/2 in nm plus a constant per species); `kinetic`, `kinetic_map` and
`kinetic_vol` (a → b → c hard model on `C`, HalfNormal spectra; default inits, MAP inits, and
MAP inits plus the volume correction). Two rows are short of 16 seeds: `gp_learn` was stopped
at 3–4 seeds (six minutes per fit, and its pattern — R-hat > 2, ℓ drifting — was already the
fixed-ℓ GP's without the convergence), and every other row has its full 16 (truth mode) or 8 (prior mode) seeds; the `n`
column states the count for every row.

### B.2 Evidence tables

<!-- tables:start -->
#### Truth mode, realistic set (16 noise draws)

| approach | n | cov C | cov C a/b/c | cov S | cov S a/b/c | width C (uM) | width S (AU/uM) | R-hat max | R-hat>1.05 | div | sigma (true 0.002) | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| halfnormal | 16 | 0.741 | [0.76, 0.67, 0.80] | 0.566 | [0.84, 0.15, 0.71] | 0.656 | 0.00027 | 1.18 | 10/16 | 945 | 0.00201 | 6 |
| halfnormal_vol | 16 | 0.606 | [0.48, 0.51, 0.83] | 0.520 | [0.84, 0.09, 0.63] | 0.696 | 0.00025 | 1.15 | 11/16 | 278 | 0.00201 | 14 |
| exponential | 16 | 0.757 | [0.77, 0.66, 0.84] | 0.564 | [0.84, 0.17, 0.68] | 0.583 | 0.00027 | 1.18 | 14/16 | 143 | 0.00201 | 8 |
| horseshoe | 16 | 0.417 | [0.25, 0.82, 0.17] | 0.633 | [0.86, 0.45, 0.60] | 0.249 | 0.00023 | 2.17 | 16/16 | 0 | 0.00201 | 64 |
| rw2 | 16 | 0.455 | [0.91, 0.24, 0.21] | 0.458 | [0.81, 0.29, 0.27] | 0.178 | 0.00014 | 4.18 | 16/16 | 0 | 0.00202 | 71 |
| rw2_loose | 16 | 0.313 | [0.34, 0.49, 0.11] | 0.573 | [0.90, 0.23, 0.59] | 0.375 | 0.00016 | 239.50 | 16/16 | 8 | 0.00285 | 70 |
| gp_softplus | 16 | 0.511 | [0.33, 0.99, 0.22] | 0.700 | [0.89, 0.57, 0.64] | 0.269 | 0.00017 | 1.08 | 3/16 | 0 | 0.00200 | 71 |
| gp_log | 16 | 0.769 | [0.57, 0.97, 0.77] | 0.688 | [0.84, 0.46, 0.76] | 0.337 | 0.00018 | 1.10 | 4/16 | 0 | 0.00200 | 66 |
| gp_learn | 4 | 0.506 | [0.83, 0.23, 0.46] | 0.556 | [0.84, 0.47, 0.36] | 0.262 | 0.00015 | 2.04 | 4/4 | 0 | 0.00201 | 354 |
| bspline_exp | 16 | 0.731 | [0.77, 0.55, 0.87] | 0.487 | [0.79, 0.11, 0.56] | 0.636 | 0.00020 | 1.10 | 6/16 | 434 | 0.00201 | 12 |
| pspline | 16 | 0.287 | [0.43, 0.38, 0.06] | 0.550 | [0.87, 0.33, 0.45] | 0.271 | 0.00013 | 14.63 | 16/16 | 56 | 0.00351 | 65 |
| bands_nm | 16 | 0.876 | [0.96, 0.85, 0.82] | 0.841 | [0.87, 0.83, 0.83] | 0.144 | 0.00003 | 1.07 | 2/16 | 13388 | 0.00316 | 60 |
| bands_wn | 16 | 0.399 | [0.14, 0.97, 0.08] | 0.058 | [0.03, 0.10, 0.04] | 0.252 | 0.00006 | 1.03 | 0/16 | 8725 | 0.00357 | 72 |
| bands_free3 | 16 | 0.950 | [0.95, 0.96, 0.94] | 0.970 | [0.96, 0.99, 0.96] | 0.154 | 0.00006 | 4.96 | 16/16 | 138 | 0.00200 | 73 |
| bands_nm_off | 16 | 0.941 | [0.97, 0.94, 0.92] | 0.653 | [0.75, 0.42, 0.80] | 0.146 | 0.00005 | 1.18 | 2/16 | 14909 | 0.00252 | 40 |
| kinetic | 16 | 0.154 | [0.00, 0.00, 0.46] | 0.680 | [0.83, 0.36, 0.85] | 0.086 | 0.00020 | 1.03 | 0/16 | 0 | 0.00201 | 22 |
| kinetic_map | 16 | 0.151 | [0.00, 0.00, 0.45] | 0.678 | [0.83, 0.36, 0.85] | 0.085 | 0.00021 | 1.02 | 0/16 | 0 | 0.00201 | 22 |
| kinetic_vol | 16 | 0.136 | [0.00, 0.00, 0.41] | 0.667 | [0.83, 0.33, 0.84] | 0.089 | 0.00021 | 1.02 | 0/16 | 0 | 0.00201 | 23 |
| remix_flat | 16 | 0.932 | [0.98, 0.82, 1.00] | 0.829 | [0.95, 0.59, 0.94] | 1.271 | 0.00045 | 1.00 | 0/16 | 0 | 0.00200 | 13 |
| remix_zero | 16 | 0.996 | [1.00, 0.99, 1.00] | 0.952 | [0.93, 0.95, 0.97] | 1.097 | 0.00050 | 1.00 | 0/16 | 0 | 0.00200 | 13 |
| remix_band | 16 | 1.000 | [1.00, 1.00, 1.00] | 1.000 | [1.00, 1.00, 1.00] | 2.036 | 0.00059 | 1.00 | 0/16 | 0 | 0.00200 | 13 |
| remix_flat_sg5 | 16 | 0.955 | [0.99, 0.88, 1.00] | 0.812 | [0.88, 0.61, 0.95] | 1.308 | 0.00032 | 1.00 | 0/16 | 0 | 0.00200 | 13 |
| remix_zero_sg5 | 16 | 0.997 | [1.00, 0.99, 1.00] | 0.933 | [0.86, 0.98, 0.96] | 1.083 | 0.00037 | 1.00 | 0/16 | 0 | 0.00200 | 13 |
| remix_band_sg5 | 16 | 1.000 | [1.00, 1.00, 1.00] | 0.989 | [0.97, 1.00, 1.00] | 2.051 | 0.00046 | 1.00 | 0/16 | 0 | 0.00200 | 13 |
| remix_flat_sg7 | 16 | 0.946 | [0.99, 0.85, 1.00] | 0.771 | [0.79, 0.58, 0.95] | 1.302 | 0.00027 | 1.00 | 0/16 | 0 | 0.00200 | 13 |
| remix_zero_sg7 | 16 | 0.993 | [0.98, 0.99, 1.00] | 0.895 | [0.76, 0.97, 0.95] | 1.109 | 0.00032 | 1.00 | 0/16 | 0 | 0.00200 | 13 |
| remix_band_sg7 | 16 | 1.000 | [1.00, 1.00, 1.00] | 0.960 | [0.88, 1.00, 1.00] | 2.072 | 0.00041 | 1.00 | 0/16 | 0 | 0.00200 | 13 |

#### Truth mode, harder set (16 noise draws)

| approach | n | cov C | cov C a/b/c | cov S | cov S a/b/c | width C (uM) | width S (AU/uM) | R-hat max | R-hat>1.05 | div | sigma (true 0.002) | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| halfnormal | 16 | 0.210 | [0.09, 0.43, 0.11] | 0.678 | [0.95, 0.80, 0.29] | 1.110 | 0.00064 | 1.32 | 13/16 | 315 | 0.00200 | 9 |
| halfnormal_vol | 16 | 0.812 | [0.69, 1.00, 0.74] | 0.911 | [0.95, 0.99, 0.80] | 2.587 | 0.00055 | 1.35 | 15/16 | 263 | 0.00200 | 38 |
| exponential | 16 | 0.341 | [0.01, 0.84, 0.17] | 0.636 | [0.94, 0.52, 0.45] | 1.286 | 0.00065 | 1.23 | 14/16 | 182 | 0.00200 | 17 |
| horseshoe | 16 | 0.040 | [0.00, 0.07, 0.05] | 0.510 | [0.95, 0.34, 0.24] | 0.283 | 0.00053 | 4.85 | 16/16 | 0 | 0.00200 | 63 |
| rw2 | 16 | 0.158 | [0.11, 0.27, 0.09] | 0.682 | [0.95, 0.76, 0.34] | 0.269 | 0.00030 | 9.29 | 16/16 | 0 | 0.00199 | 71 |
| rw2_loose | 16 | 0.052 | [0.01, 0.10, 0.05] | 0.532 | [0.90, 0.46, 0.23] | 0.235 | 0.00034 | 17.63 | 16/16 | 0 | 0.00201 | 69 |
| gp_softplus | 16 | 0.034 | [0.00, 0.06, 0.04] | 0.508 | [0.95, 0.33, 0.24] | 0.285 | 0.00049 | 2.54 | 8/16 | 0 | 0.00200 | 71 |
| gp_log | 16 | 0.038 | [0.00, 0.07, 0.04] | 0.511 | [0.95, 0.34, 0.25] | 0.365 | 0.00056 | 1.99 | 13/16 | 0 | 0.00200 | 67 |
| gp_learn | 3 | 0.851 | [0.79, 1.00, 0.76] | 0.944 | [0.96, 0.97, 0.90] | 0.990 | 0.00037 | 2.42 | 3/3 | 0 | 0.00199 | 351 |
| bspline_exp | 16 | 0.652 | [0.14, 0.85, 0.97] | 0.845 | [0.94, 0.76, 0.84] | 1.555 | 0.00054 | 1.12 | 4/16 | 100 | 0.00200 | 32 |
| pspline | 16 | 0.080 | [0.05, 0.13, 0.05] | 0.498 | [0.89, 0.37, 0.23] | 0.252 | 0.00031 | 48.14 | 16/16 | 35 | 0.00290 | 65 |
| bands_nm | 16 | 0.295 | [0.85, 0.00, 0.03] | 0.054 | [0.09, 0.02, 0.06] | 0.974 | 0.00044 | 1.02 | 0/16 | 0 | 0.01480 | 16 |
| bands_wn | 16 | 0.159 | [0.44, 0.02, 0.03] | 0.073 | [0.14, 0.03, 0.05] | 1.037 | 0.00040 | 1.02 | 0/16 | 0 | 0.01350 | 22 |
| bands_free3 | 16 | 0.465 | [0.37, 0.48, 0.55] | 0.642 | [0.78, 0.52, 0.63] | 0.191 | 0.00010 | 72.08 | 5/16 | 156 | 0.00210 | 70 |
| bands_nm_off | 16 | 0.942 | [0.97, 0.93, 0.92] | 0.921 | [0.94, 0.88, 0.94] | 0.151 | 0.00008 | 69.82 | 3/16 | 15596 | 0.00277 | 40 |
| kinetic | 16 | 0.727 | [0.75, 0.67, 0.76] | 0.910 | [0.95, 0.86, 0.92] | 0.145 | 0.00022 | 1.05 | 1/16 | 3 | 0.00200 | 33 |
| kinetic_map | 16 | 0.740 | [0.75, 0.71, 0.76] | 0.907 | [0.95, 0.86, 0.91] | 0.148 | 0.00022 | 1.08 | 2/16 | 1 | 0.00200 | 28 |
| kinetic_vol | 16 | 0.944 | [0.94, 0.94, 0.96] | 0.942 | [0.95, 0.95, 0.93] | 0.139 | 0.00022 | 1.14 | 1/16 | 0 | 0.00200 | 22 |
| remix_flat | 16 | 0.994 | [0.98, 1.00, 1.00] | 0.965 | [0.95, 0.99, 0.95] | 3.187 | 0.00059 | 1.00 | 0/16 | 0 | 0.00200 | 13 |
| remix_zero | 16 | 0.986 | [0.97, 1.00, 0.98] | 0.961 | [0.95, 0.99, 0.94] | 3.356 | 0.00065 | 1.00 | 0/16 | 0 | 0.00200 | 13 |
| remix_band | 16 | 1.000 | [1.00, 1.00, 1.00] | 1.000 | [1.00, 1.00, 1.00] | 4.484 | 0.00120 | 1.00 | 0/16 | 0 | 0.00200 | 13 |
| remix_flat_sg5 | 16 | 0.991 | [0.97, 1.00, 1.00] | 0.938 | [0.88, 0.99, 0.95] | 3.183 | 0.00050 | 1.00 | 0/16 | 0 | 0.00200 | 13 |
| remix_zero_sg5 | 16 | 0.991 | [0.97, 1.00, 1.00] | 0.935 | [0.88, 0.99, 0.94] | 3.355 | 0.00055 | 1.00 | 0/16 | 0 | 0.00200 | 13 |
| remix_band_sg5 | 16 | 1.000 | [1.00, 1.00, 1.00] | 0.989 | [0.97, 1.00, 1.00] | 4.512 | 0.00094 | 1.00 | 0/16 | 0 | 0.00200 | 13 |
| remix_flat_sg7 | 16 | 0.993 | [0.98, 1.00, 1.00] | 0.898 | [0.79, 0.99, 0.92] | 3.197 | 0.00048 | 1.00 | 0/16 | 0 | 0.00200 | 13 |
| remix_zero_sg7 | 16 | 0.994 | [0.98, 1.00, 1.00] | 0.898 | [0.79, 0.98, 0.92] | 3.366 | 0.00053 | 1.00 | 0/16 | 0 | 0.00200 | 13 |
| remix_band_sg7 | 16 | 1.000 | [1.00, 1.00, 1.00] | 0.960 | [0.88, 1.00, 1.00] | 4.495 | 0.00086 | 1.00 | 0/16 | 0 | 0.00200 | 13 |

#### Prior mode: truth drawn from each approach's own prior (8 draws; the calibration check)

| approach | n | cov C | cov C a/b/c | cov S | cov S a/b/c | width C (uM) | width S (AU/uM) | R-hat max | R-hat>1.05 | div | sigma (true 0.002) | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| halfnormal | 8 | 0.990 | [0.98, 1.00, 0.99] | 0.968 | [0.96, 0.98, 0.97] | 0.523 | 0.00265 | 1.11 | 4/8 | 69 | 0.00817 | 7 |
| halfnormal_vol | 8 | 0.974 | [0.98, 0.97, 0.97] | 0.949 | [0.96, 0.94, 0.95] | 0.479 | 0.00246 | 1.24 | 4/8 | 93 | 0.00817 | 15 |
| exponential | 8 | 0.958 | [0.95, 0.96, 0.97] | 0.958 | [0.96, 0.95, 0.96] | 0.608 | 0.00106 | 1.10 | 2/8 | 229 | 0.00816 | 5 |
| horseshoe | 8 | 0.642 | [0.51, 0.76, 0.66] | 0.833 | [0.92, 0.80, 0.78] | 0.081 | 0.00102 | 164.48 | 8/8 | 0 | 0.00871 | 62 |
| rw2 | 8 | 0.372 | [0.41, 0.41, 0.30] | 0.676 | [0.94, 0.54, 0.54] | 0.365 | 0.00068 | 25.23 | 7/8 | 0 | 0.00687 | 72 |
| rw2_loose | 8 | 0.270 | [0.41, 0.20, 0.20] | 0.564 | [0.82, 0.47, 0.41] | 0.182 | 0.00039 | 71.59 | 7/8 | 293 | 0.00716 | 69 |
| gp_softplus | 8 | 0.917 | [0.96, 0.86, 0.93] | 0.954 | [0.96, 0.95, 0.96] | 0.228 | 0.00090 | 2.40 | 2/8 | 0 | 0.00813 | 66 |
| gp_log | 8 | 0.826 | [0.87, 0.71, 0.90] | 0.903 | [0.96, 0.83, 0.92] | 0.096 | 0.00093 | 12.44 | 7/8 | 0 | 0.00813 | 62 |
| gp_learn | 3 | 1.000 | [1.00, 1.00, 1.00] | 0.965 | [0.92, 0.98, 0.99] | 0.239 | 0.00110 | 1.30 | 1/3 | 0 | 0.00654 | 359 |
| bspline_exp | 8 | 0.938 | [0.94, 0.91, 0.97] | 0.953 | [0.96, 0.94, 0.96] | 1.317 | 0.00075 | 1.03 | 0/8 | 8 | 0.00810 | 7 |
| pspline | 8 | 0.358 | [0.53, 0.33, 0.22] | 0.630 | [0.93, 0.59, 0.37] | 0.426 | 0.00013 | 57.27 | 8/8 | 551 | 0.00692 | 71 |
| bands_nm | 8 | 0.964 | [0.96, 0.96, 0.97] | 0.977 | [0.96, 1.00, 0.97] | 0.461 | 0.00003 | 1.11 | 1/8 | 6580 | 0.00686 | 25 |
| bands_wn | 8 | 0.908 | [0.97, 0.90, 0.85] | 0.931 | [0.98, 0.90, 0.91] | 0.505 | 0.00004 | 1.08 | 2/8 | 8066 | 0.00687 | 31 |
| bands_free3 | 8 | 0.879 | [0.94, 0.82, 0.88] | 0.814 | [0.87, 0.67, 0.90] | 0.496 | 0.00033 | 2.05 | 3/8 | 328 | 0.00755 | 43 |
| bands_nm_off | 8 | 0.964 | [0.96, 0.97, 0.96] | 0.915 | [0.84, 1.00, 0.91] | 0.615 | 0.00046 | 1.06 | 1/8 | 6255 | 0.01033 | 38 |
| kinetic | 8 | 0.918 | [1.00, 0.87, 0.88] | 0.953 | [0.95, 0.96, 0.95] | 0.095 | 0.00094 | 1.02 | 0/8 | 0 | 0.00814 | 25 |
| kinetic_map | 8 | 0.864 | [1.00, 0.77, 0.82] | 0.956 | [0.96, 0.96, 0.95] | 0.096 | 0.00095 | 1.02 | 0/8 | 0 | 0.00814 | 31 |
| kinetic_vol | 8 | 0.746 | [0.75, 0.61, 0.88] | 0.877 | [0.96, 0.73, 0.95] | 0.102 | 0.00100 | 1.02 | 0/8 | 0 | 0.00814 | 24 |
| remix_flat | 8 | 0.968 | [0.95, 0.96, 1.00] | 0.958 | [0.95, 0.94, 0.98] | 1.219 | 0.00045 | 1.00 | 0/8 | 0 | 0.00200 | 12 |
| remix_zero | 8 | 0.901 | [0.93, 0.91, 0.86] | 0.910 | [0.95, 0.82, 0.96] | 1.059 | 0.00049 | 1.00 | 0/8 | 0 | 0.00200 | 12 |
| remix_band | 8 | 1.000 | [1.00, 1.00, 1.00] | 1.000 | [1.00, 1.00, 1.00] | 1.998 | 0.00062 | 1.00 | 0/8 | 0 | 0.00200 | 12 |
| remix_flat_sg5 | 8 | 0.967 | [0.97, 0.93, 1.00] | 0.879 | [0.82, 0.88, 0.94] | 1.253 | 0.00033 | 1.00 | 0/8 | 0 | 0.00200 | 12 |
| remix_zero_sg5 | 8 | 0.895 | [0.92, 0.91, 0.85] | 0.834 | [0.80, 0.78, 0.91] | 1.056 | 0.00038 | 1.00 | 0/8 | 0 | 0.00200 | 12 |
| remix_band_sg5 | 8 | 1.000 | [1.00, 1.00, 1.00] | 0.986 | [0.96, 1.00, 1.00] | 2.032 | 0.00049 | 1.00 | 0/8 | 0 | 0.00200 | 12 |
| remix_flat_sg7 | 8 | 0.957 | [0.96, 0.91, 1.00] | 0.805 | [0.67, 0.85, 0.90] | 1.238 | 0.00027 | 1.00 | 0/8 | 0 | 0.00200 | 13 |
| remix_zero_sg7 | 8 | 0.899 | [0.92, 0.90, 0.88] | 0.754 | [0.65, 0.74, 0.87] | 1.057 | 0.00031 | 1.00 | 0/8 | 0 | 0.00200 | 13 |
| remix_band_sg7 | 8 | 1.000 | [1.00, 1.00, 1.00] | 0.946 | [0.85, 1.00, 0.99] | 2.022 | 0.00044 | 1.00 | 0/8 | 0 | 0.00200 | 13 |


#### Prior mode: posterior σ / true σ

- halfnormal: 1.001
- halfnormal_vol: 1.001
- exponential: 1.001
- horseshoe: 1.252
- rw2: 0.999
- rw2_loose: 1.132
- gp_softplus: 0.998
- gp_log: 0.998
- gp_learn: 1.006
- bspline_exp: 0.995
- pspline: 1.014
- bands_nm: 0.996
- bands_wn: 0.996
- bands_free3: 1.117
- bands_nm_off: 0.996
- kinetic: 0.998
- kinetic_map: 0.998
- kinetic_vol: 0.998
- remix_flat: 0.998
- remix_zero: 0.998
- remix_band: 0.998
- remix_flat_sg5: 0.998
- remix_zero_sg5: 0.998
- remix_band_sg5: 0.998
- remix_flat_sg7: 0.998
- remix_zero_sg7: 0.998
- remix_band_sg7: 0.998
<!-- tables:end -->

### B.3 Reading the tables

**σ first.** The posterior σ equals the true 0.002 to three decimals in every row except the
ones where it flags a wrong answer: the band model in its wrong optimum (2 of 16 seeds at
σ = 0.011, which pulls the mean to 0.0032), the band-plus-offset model in its wrong-mode seeds
(0.0025–0.0028), the band model on the offset data (0.0148: a constant is not a band), the
wavenumber band model (0.0036 and 0.0135: an 8% skew is visible at this noise), and the
unconverged loose random walk and P-spline (0.0029; 0.0025–0.0037). σ is therefore a necessary
check and not a sufficient one — every entry-wise and smoothness description fits the
data at exactly the noise level and is still wrong about the split, which is what partial
identification means.

**Prior mode, the calibration check.** Under their own priors, `halfnormal` (0.99 / 0.97),
`exponential` (0.96 / 0.96), `halfnormal_vol` (0.97 / 0.95), `bands_nm` (0.96 / 0.98),
`bands_nm_off` (0.96 / 0.92), `kinetic` (0.92 / 0.95), `gp_softplus` (0.92 / 0.95),
`bspline_exp` (0.94 / 0.95), `remix_flat` (0.97 / 0.96) and `remix_band` (1.00 / 1.00) are
calibrated to the resolution of 8 draws: the code, the scoring and the samplers are right for
them. Four descriptions fail their own check — `horseshoe` (0.64 / 0.83, R-hat 164), `rw2`
(0.37 / 0.68, R-hat 25), `rw2_loose` (0.27 / 0.56, R-hat 72) and `pspline` (0.36 / 0.63, R-hat 57)
— and `gp_log` is marginal (0.83 / 0.90, R-hat 12): their samplers do not mix even on data their
own prior generated, so their truth-mode rows measure the sampler as much as the prior.
`remix_zero` under-covers under the flat prior (0.90 / 0.91): the zero preference is an
informative prior, and that is its stated cost.

**Truth mode, harder set — the clean test of the split.** Every entry-wise and smoothness
description fails on the amounts: `halfnormal` 0.21, `exponential` 0.34, `horseshoe` 0.04,
`rw2` 0.16, `rw2_loose` 0.05, `gp_softplus` 0.03, `gp_log` 0.04, `pspline` 0.08,
`bspline_exp` 0.66. The two overlapping two-band species a and c are the casualties (0.00–0.14
for most rows); the one-band species b, which the data pin best, survives (0.06–0.86). Interval
widths are a fifth to a third of the band. The tilt table (A.5) predicted this: the half-normal's
0.18 predicted against 0.21 measured, and the smoothness priors, which concentrate the split on
≤ 1% of `Θ`, land at 0.03–0.16. The parametric band model fails for a different reason
(misspecification, σ 0.0148), and the band-plus-offset model, whose family contains the truth,
covers 0.94 / 0.92 with intervals 0.15 µM wide — thirty times narrower than the band — because it
identifies. The two corrected models move as A.3 says: `halfnormal_vol` 0.21 → 0.81 on amounts
and 0.68 → 0.91 on spectra, `kinetic_vol` 0.73 → 0.94 / 0.94. The re-mixing intervals are
calibrated — flat 0.99 / 0.97, zero-weighted 0.99 / 0.96, band 1.00 / 1.00 — at widths of 3.2,
3.4 and 4.5 µM, which is the price of honesty on data whose ambiguity is genuinely that large
(a and c overlap at cosine 0.47 and nothing is selective).

**Truth mode, realistic set — the boundary.** The truth has 21% exact zeros, and two things
change. Coverage of the spectra is capped for every positive-support description (0.46–0.70
across the NUTS rows; species b, with 44% zeros, sits at 0.09–0.57), so those spectral numbers
measure the boundary, not the split. And the split itself behaves differently: `halfnormal`,
`exponential` and `bspline_exp` reach 0.73–0.76 on the amounts, because the volume factor happens
to favour the corner of `Θ` where this truth sits (A.5: truth rank 0.02) — the same descriptions
collapsed to 0.2–0.3 on the harder set. That agreement is luck, not calibration, and it reverses
as soon as the truth moves into the interior. The smoothness priors stay bad (`rw2` 0.46,
`gp_softplus` 0.51, `pspline` 0.29) because a walk or a GP through a link cannot sit on a zero.
The band model in its right optimum covers 0.94 / 0.93 (14 of 16 seeds). The flat re-mixing
interval under-covers exactly where Moon & Schorfheide say it must — a truth on the boundary of
`Θ` is in the outer 2.5% of a flat prior for many values (b: 0.82 on amounts, 0.59 on its
spectrum) — the zero-weighted summary recovers it (1.00 / 0.95), and the band covers everything
(1.00 / 1.00).

**Convergence.** Every NUTS description with free spectra has a 4-dimensional nearly flat
ridge in its posterior. R-hat exceeds 1.05 in 10–16 of 16 fits for `halfnormal`, `exponential`,
`horseshoe`, `rw2`, `rw2_loose` and `pspline` (maxima 1.2 to 240), with ESS in the tens or single
digits, despite MAP starts and 3 000 steps per chain. The GP with fixed ℓ mixes best on the realistic set (R-hat ≤ 1.05 in 12–13 of 16) and only
half the time on the harder set (8 and 13 of 16 above 1.05). The band models mix (R-hat 1.02–1.07 in the right optimum) but throw 500–1 000
divergences per fit — the (μ, w, h) posterior is stiff — and `bands_free3` shows R-hat 5–72
because its spare bands swap within a species; its pooled intervals still cover (0.95 / 0.97 on
the realistic set) because the swap is a symmetry. The re-mixing method has no sampler to
converge: hit-and-run over a 4-dimensional convex region, 3 750 draws, 13 s per dataset including
both smoothed variants.

**Wall clock**, under seven-way contention: re-mixing 13 s for all interval kinds; half-normal
and exponential 6–17 s; band models 40–70 s plus 5–30 s of multi-start; GP, RW2 and P-spline
65–75 s; learnable-ℓ GP 360 s (a Cholesky per gradient with ℓ moving, and the worst mixing of
the GPs); kinetic 22–33 s.

### B.4 The kinetic row, explained

Finding 4 of the brief — a hard a → b → c model on `C` converges in seconds but its intervals
miss — is reproduced and explained.

| set | n | k₁ range over seeds (true 0.60) | k₂ range (true 0.40) | amount coverage | a / b / c | seeds fully covered | R-hat max | σ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| realistic | 16 | 0.616–0.634 | 0.378–0.389 | 0.15 | 0.00 / 0.00 / 0.46 | 0/16 | 1.03 | 0.00201 |
| harder | 16 | 0.540–0.607 | 0.395–0.447 | 0.73 | 0.75 / 0.67 / 0.76 | 10/16 | 1.05 | 0.00200 |

Both chains agree in every seed, no chain ever swaps the two rates (0 of 32
fits), σ is exact, and on the realistic set k₁ comes out 3–5% high and k₂ 4% low with a
posterior width of about 0.01 — a bias ten times the width, identical across seeds. That is not
a sampler problem and not the rate-swap ambiguity of consecutive first-order steps; it is the
free spectra.

Profiling the likelihood over (k₁, k₂) with the spectra fitted by least squares at every grid
point (`kprofile.py`, seed 0) shows why:

- With **unconstrained** spectra the profile is a plateau: its maximum is at (0.603, 0.398) and
  its 95% range is k₁ 0.573–0.630, k₂ 0.382–0.420 — the truth loses 0.7 nats against the
  maximum. The column space of the data pins the two decay rates only weakly once 192 spectrum
  values are free to absorb a small change in the kinetic shapes. Any prior or geometry term that
  varies by a few nats across that plateau moves the posterior by several percent in k.
- The Occam term of A.3 — here only the spectra are free coordinates, so
  `V(k) ∝ det(C(k)ᵀC(k) + rᵀr)^(−n_w/2)` — varies by 1.7 nats across the plateau and moves the
  optimum to (0.580, 0.413). That is where the harder-set posteriors sit (k₁ 0.55–0.60 across
  seeds): interior truth, no boundary, the Occam term decides, and the intervals cover in
  10 of 16 seeds and miss in the rest.
- On the realistic set the **non-negativity** of the spectra binds: 21% of the true values are
  exact zeros, so half of the compensating least-squares spectra would go negative. The
  constrained profile (exact, all eight active sets per channel) has its 95% range at k₁
  0.600–0.628, k₂ 0.383–0.402 — the truth sits at the *edge* of what the constraint leaves, and
  the posterior, which lives in the interior of that cut plateau, lands at (0.62, 0.385) with a
  width of 0.01 in every seed. Coverage for a and b is exactly zero, as measured.

So a hard kinetic model on `C` does identify the split — but with *free* spectra it inherits
both hidden opinions of A.3, the Occam term and the boundary, and its narrow intervals report
those, not the data. `kinetic_map` (chains started at the multi-start MAP) reproduces the same
numbers, so initialisation is not the cause. Adding the volume correction (`kinetic_vol`,
`+½ n_w log det(CᵀC + rᵀr)`) removes the Occam term; restores amount and spectrum coverage to 0.94 / 0.94 on the harder set (σ 0.00200, R-hat ≤ 1.14, 16 seeds; uncorrected 0.73 / 0.91) and,
as expected, does nothing for the boundary bias on the realistic set (k₁ = 0.63, coverage
0.14 over 16 seeds). For the project's next stage the lesson is
concrete: a kinetic model does not make the spectra a nuisance that can be marginalised with a
flat prior. Either profile the spectra out (the volume correction is that, in closed form),
or give them the explicit low-rank parametrisation of C.1, where the kinetic model becomes a
constraint on `T` and the ambiguity collapses without any marginalisation.

### B.5 Smoothing along wavelength before pattern extraction

The question was whether smoothing the data along wavelength before the SVD narrows the
noise margin without touching the ambiguity part. Measured on the re-mixing method with a
local-quadratic (Savitzky–Golay-type) filter of 5 and 7 channels, 16 seeds per dataset,
median over all values:

| dataset | window | band width S | ambiguity part S | noise margin S | band width C | ambiguity part C | noise margin C | residual σ after smoothing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| realistic | none | 0.00060 | 0.00017 | 0.00036 | 2.04 | 1.74 | 0.28 | 0.00200 |
| realistic | 5 | 0.00047 | 0.00017 | 0.00025 | 2.06 | 1.77 | 0.28 | 0.00139 |
| realistic | 7 | 0.00042 | 0.00017 | 0.00021 | 2.08 | 1.78 | 0.28 | 0.00114 |
| harder | none | 0.00119 | 0.00049 | 0.00047 | 4.50 | 4.27 | 0.28 | 0.00200 |
| harder | 5 | 0.00095 | 0.00048 | 0.00033 | 4.52 | 4.28 | 0.28 | 0.00139 |
| harder | 7 | 0.00086 | 0.00049 | 0.00027 | 4.51 | 4.26 | 0.28 | 0.00114 |

(S in AU/µM, C in µM; "band width" = ambiguity part + margin; the region is always defined
with the *raw* noise slack, so only the margin changes.)

Three things are true at once:

1. **The noise margin on the spectra shrinks by exactly the filter's noise-reduction factor**
   — 0.70 for 5 channels, 0.57 for 7 — and the ambiguity part does not move (0.00017 →
   0.00017; 0.00049 → 0.00048), nor does anything on the amounts: the scores' noise is the
   projection of the noise onto the pattern directions, which a wavelength filter does not
   remove (margin 0.28 µM throughout). So the answer to the question as asked is *yes, for
   the spectra, and not at all for the amounts*.
2. **The narrowing is paid for in bias that the margin does not know about.** Species a has
   its strong band at 355 nm with σ = 16 nm, i.e. 2.8 channels; a 7-channel quadratic clips
   that peak. The band's coverage of the true spectra drops from 1.000 to 0.989 (5 channels)
   and 0.960 (7 channels) on both datasets, and for species a alone to 0.97 and 0.88; the
   flat interval's spectral coverage drops from 0.83 to 0.81 and 0.77 (realistic) and from
   0.965 to 0.938 and 0.898 (harder). The amounts are unaffected (all ≥ 0.99 for the band).
3. **The margin was never the problem.** On the realistic set the ambiguity part is a third of
   the spectral band and the noise margin two thirds, so smoothing looks attractive there;
   on the harder set (no zero regions to pin the region) the ambiguity part is already the
   larger half, and on real data with more overlap it dominates. Smoothing narrows the part
   that was honest already.

Rule that follows: smooth only with a window below the narrowest band's σ in channels, or
add the smoother's bias (its response to the fitted pattern minus the pattern, computable)
to the margin. The clean version of the same idea, not tested here, is to smooth the *span*
rather than the data — fit each pattern of `Y` with a penalty chosen by cross-validation,
then re-mix — which keeps the ambiguity untouched by construction and puts the bias where it
can be estimated.

---

## C. Recommendation

### C.1 The description to use

Describe the pure spectra as **re-mixings of the data's own rank-K patterns**, `S = T⁻¹·Y`
and `C = X·T`, with the split `T` confined to the feasible region `Θ` by non-negativity, closure
and every known reference, a **flat prior on the free re-mixing numbers**, and the measurement
noise propagated per draw (description iv, `remix.py`). Report the **band** — `Θ` projected onto
each value, ± the propagated noise margin — as the 95% interval, and the flat posterior's central
interval as the typical-solution summary, labelled as such. Reasons, in the order the evidence
gives them:

1. **It is the only description whose interval contained the truth at least 95% of the time on
   both datasets, in truth mode and under its own prior.** Band 1.00 / 1.00 everywhere; flat
   0.99 / 0.97 on the interior truth and 0.97 / 0.96 under its prior. Every entry-wise and
   smoothness prior covered 3–66% of the amounts on the interior truth, each with σ exact.
2. **Its opinion about the split is explicit, flat and visible**, and the likelihood's volume
   factor — the dominant hidden opinion of every other description, 13–22 nats across `Θ` — is
   constant for it because the noise directions are not parameters. It is the one description
   for which calibration is a theorem (the robust credible region of Giacomini & Kitagawa 2021)
   rather than an observation.
3. **It has no sampler to converge**: 13 s per dataset for all interval kinds, no R-hat, no
   divergences, against 3 000 NUTS steps that still leave R-hat above 1.05 in most fits of every
   free-spectra model.
4. **Its width is the width of the ambiguity**: 2.0 µM on the realistic set and 4.5 µM on the
   harder set for the amounts. No description that reports less is reporting the data; the ones
   that did were 5–30× narrower and wrong.

Two qualified companions, not alternatives:

- **A parametric band model** is the right description when the chemistry says the spectra are
  sums of a known number of bands *and the family is right*. It identifies rather than
  regularises, and its intervals are then 14–30× narrower than the band and calibrated (0.94 /
  0.93 in the right optimum, 14 of 16 seeds). It is confidently wrong under misspecification
  (Gaussians in wavenumber: spectral coverage 0.05; a 0.003 AU/µM offset: σ seven times too
  high), needs a 32-start optimiser, and still lands in the wrong mode in 2 of 16 seeds, where
  only σ gives it away. Use it as a hypothesis test against the band — if its posterior lies
  inside the band and σ matches the noise, the chemistry is consistent; if not, the family is
  wrong — never as the default.
- **For the kinetic stage**: a kinetic model on `C` with free non-negative spectra is not
  calibrated by itself (0.15 and 0.73 on the two sets, B.4). With the volume correction it
  reaches 0.94 / 0.94 on the interior truth and still fails at exact zeros. The cleaner route
  is to apply the kinetic model as a constraint on `T` inside description (iv): the profiles
  become `X·T` with `T` pinned by the rate law, `Θ` collapses, and nothing is marginalised.

What this changes in the plan: the v0 random walk is out as a description of `S` (R-hat 4–25,
coverage 0.16–0.46, and 1 000–4 000 nats of opinion on the split); step 4 (smoothness on `C`,
sampled τ) is moot; the feasible-band calculation that principle 6 asks for as a check becomes
the primary output; and a `numpyro.factor` with `+½ log pdet(JᵀJ)` — one line, closed form in
A.3 — belongs in any model that keeps free spectra next to anything identifying.

### C.2 What it does not handle

- **A single narrow interval on a boundary truth.** When the true spectra have exact zeros the
  truth sits on the boundary of `Θ`, and any interval narrower than the band — flat, zero-weighted
  or NUTS — under-covers there by construction (Moon & Schorfheide). The band is the only interval
  with a guarantee; the flat and zero-weighted summaries are typical-solution summaries with a stated
  prior. Report the band; quote the summaries as such.
- **Unknown rank.** `K` is given. A fourth, absent species (the `extra.py` question of the earlier
  session) is not addressed here; with `K` too large the region gains dimensions and the band widens
  to uselessness rather than shrinking the spare species away.
- **Many components.** The free numbers scale as `K² − (equality facts)`: 4 here, 2 with two reference
  scans, but ~20 for five species with one scan. Hit-and-run in 4 dimensions is cheap (13 s including
  the 8 000-step chain); in 20 it needs a proper AFS algorithm (polygon inflation, Sawall et al. 2013)
  and the band becomes the only usable output.
- **Structured noise.** The margin assumes the residual after the rank-3 projection is white with one
  σ. Wavelength-dependent noise, absorbance-dependent noise, drift and baselines all violate that;
  a baseline in particular *adds a component* and shows up as rank 4. The NUTS descriptions can model
  those (σ(λ), a baseline term); the re-mixing method as implemented cannot, and would need the
  patterns extracted under a weighted or augmented model first.
- **Non-negativity as the only shape constraint.** The region uses `C ≥ 0`, `S ≥ 0`, closure and the
  scan. Everything the field knows about narrowing `Θ` — selective windows, unimodal *profiles*,
  known spectra, several runs sharing one `S`, a kinetic model — is an extra linear or convex
  constraint on `T` that the same machinery accepts and that this bench did not use. Those are the
  levers that shrink the band; the description choice is not.
- **Smoothing of the patterns.** The patterns carry the projection's noise, unsmoothed. B.5 shows
  smoothing the data is a bias trade; smoothing the span is the right version and was not built.
- **Real instruments.** Everything here is synthetic Gaussian bands with white noise at σ = 0.002
  and absorbance ≤ 0.35. Nothing about the `tests/data/` fixtures has been tried (no reader yet,
  ADR 0002).

### C.3 The smoothing question, answered

Yes for the spectra, no for the amounts, and not for free: smoothing along wavelength before the
SVD narrows the spectral noise margin by the filter's own noise-reduction factor (0.70 at 5 channels,
0.57 at 7), leaves the ambiguity part unchanged to ±2%, leaves the amounts' margin unchanged (the
scores' noise lives in the pattern directions, which a wavelength filter does not touch), and
introduces a bias at any band narrower than the window that the margin does not contain — spectral
coverage of the band fell from 1.000 to 0.989 (5 channels) and 0.960 (7 channels), and for the
narrowest-band species to 0.88. Use it only with a window shorter than the narrowest band's σ in
channels, or add the smoother's bias to the margin; better, smooth the span (fit the patterns with a
cross-validated penalty) and leave the data alone. Full numbers in B.5.

### C.4 Side findings

- **The coverage criterion has a ceiling at exact zeros.** 21% of the realistic truth's spectrum
  values are exactly zero (44% for species b). For any description with strictly positive support
  (all the NUTS ones) the lower end of a central interval is positive, so those values can never be
  "inside"; spectral coverage on the realistic set is capped near 0.8 for the NUTS rows regardless
  of the split. The re-mixing intervals are clipped at zero and counted with `≥`, so they do not
  suffer this. The harder set (no zeros) is the clean comparison of the split; the realistic set
  tests the boundary behaviour. Both are reported.
- **The band model's bad optimum is detectable.** In the seeds where the multi-start MAP lands in
  the wrong mode, the posterior σ is 5–7× the noise. The hard rule "posterior σ must match the true
  0.002" flags every one of those fits; nothing else in the table does.
- **Two DOIs in `spec.md` §4 are wrong** (see References). Not edited here.
- **The likelihood volume factor is a general result** for bilinear models with more channels than
  timepoints (or vice versa), not specific to these priors; A.3 gives it in closed form.

---

## Reproduction

Scripts (throwaway, copied next to this report in `bench/`, where a `.gitignore` keeps the
`.py` files out of git, ty, ruff and pre-commit — they are not library code; they ran from this
session's scratchpad with `uv run python`):

- `bench/common.py` — data, noise draws, relabelling, scoring, JSONL rows.
- `bench/bench.py` — every NUTS description (`REGISTRY`), the closure amounts model, the
  hard kinetic amounts, multi-start MAP initialisation, the volume factor.
  `bench.py APPROACH DATASET truth|prior SEEDS OUT.jsonl`.
- `bench/remix.py` — patterns + re-mixing region + hit-and-run; flat, zero-weighted and band
  intervals; optional pre-smoothing. `remix.py DATASET truth|prior SEEDS OUT.jsonl`.
- `bench/tilt.py` — induced prior over Θ for each description, the likelihood volume factor,
  predicted split posteriors. `tilt.py DATASET OUT.json`.
- `bench/plots.py` — seed-0 figures (`bench/results/figures/seed0_<dataset>.png`): amounts over
  time and pure spectra for six descriptions, mode plus pointwise 90% HDI, truth dashed, the
  full feasible band dotted on the re-mixing row. Run with `uv run --with matplotlib`.
- `bench/aggregate.py` — JSONL → the tables above. `bench/runner.py` — N-at-a-time job runner;
  `bench/sweep_cmds.txt` and `bench/vol_cmds.txt` are the exact job lists.
- `bench/results/` — the aggregated tables (`tables.md`) and the tilt JSONs. The raw JSONL rows
  (one per fit, with the seed-0 intervals) stay in the session scratchpad; they are large and
  regenerable.

Machine: Apple M4 Pro (14 cores), float64, `numpyro.set_host_device_count(2)`, JAX 0.11.2,
NumPyro 0.22.0; 7 jobs in parallel, so wall clocks are under contention and comparable only
within the table.

## References

All entries verified against Crossref on 2026-09-22 (a background check run for this report).

**Rotational ambiguity and feasible bands**

- Lawton, W. H.; Sylvestre, E. A. (1971). Self modeling curve resolution. *Technometrics* 13(3), 617–633. doi:10.1080/00401706.1971.10488823
- Manne, R. (1995). On the resolution problem in hyphenated chromatography. *Chemom. Intell. Lab. Syst.* 27(1), 89–94. doi:10.1016/0169-7439(95)80009-X
- Tauler, R. (1995). Multivariate curve resolution applied to second order data. *Chemom. Intell. Lab. Syst.* 30(1), 133–146. doi:10.1016/0169-7439(95)00047-X
- Gemperline, P. J. (1999). Computation of the range of feasible solutions in self-modeling curve resolution algorithms. *Anal. Chem.* 71(23), 5398–5404. doi:10.1021/ac990648y
- Tauler, R. (2001). Calculation of maximum and minimum band boundaries of feasible solutions for species profiles obtained by multivariate curve resolution. *J. Chemom.* 15(8), 627–646. doi:10.1002/cem.654
- Rajkó, R.; István, K. (2005). Analytical solution for determining feasible regions of self-modeling curve resolution (SMCR) method based on computational geometry. *J. Chemom.* 19(8), 448–463. doi:10.1002/cem.947
- Jaumot, J.; Tauler, R. (2010). MCR-BANDS: A user friendly MATLAB program for the evaluation of rotation ambiguities in Multivariate Curve Resolution. *Chemom. Intell. Lab. Syst.* 103(2), 96–107. doi:10.1016/j.chemolab.2010.05.020
- Abdollahi, H.; Tauler, R. (2011). Uniqueness and rotation ambiguities in Multivariate Curve Resolution methods. *Chemom. Intell. Lab. Syst.* 108(2), 100–111. doi:10.1016/j.chemolab.2011.05.009
- Sawall, M.; Kubis, C.; Selent, D.; Börner, A.; Neymeyr, K. (2013). A fast polygon inflation algorithm to compute the area of feasible solutions for three-component systems. I: Concepts and applications. *J. Chemom.* 27(5), 106–116. doi:10.1002/cem.2498
- Golshan, A.; Abdollahi, H.; Beyramysoltan, S.; Maeder, M.; Neymeyr, K.; Rajkó, R.; Sawall, M.; Tauler, R. (2016). A review of recent methods for the determination of ranges of feasible solutions resulting from soft modelling analyses of multivariate data. *Anal. Chim. Acta* 911, 1–13. doi:10.1016/j.aca.2016.01.011
- de Juan, A.; Tauler, R. (2021). Multivariate Curve Resolution: 50 years addressing the mixture analysis problem – A review. *Anal. Chim. Acta* 1145, 59–78. doi:10.1016/j.aca.2020.10.051
- Olivieri, A. C.; Tauler, R. (2021). N-BANDS: A new algorithm for estimating the extension of feasible bands in multivariate curve resolution of multicomponent systems in the presence of noise and rotational ambiguity. *J. Chemom.* 35(3), e3317. doi:10.1002/cem.3317
- Olivieri, A. C. (2021). Estimating the boundaries of the feasible profiles in the bilinear decomposition of multi-component data matrices. *Chemom. Intell. Lab. Syst.* 216, 104387. doi:10.1016/j.chemolab.2021.104387
- Olivieri, A. C. (2025). Hazards of using multivariate curve resolution for processing first-order spectral data. A rotational ambiguity analysis. *Anal. Chim. Acta* 1367, 344304. doi:10.1016/j.aca.2025.344304

**Smoothing priors**

- Kimeldorf, G. S.; Wahba, G. (1970). A correspondence between Bayesian estimation on stochastic processes and smoothing by splines. *Ann. Math. Statist.* 41(2), 495–502. doi:10.1214/aoms/1177697089
- Eilers, P. H. C.; Marx, B. D. (1996). Flexible smoothing with B-splines and penalties. *Statist. Sci.* 11(2), 89–121. doi:10.1214/ss/1038425655
- Carvalho, C. M.; Polson, N. G.; Scott, J. G. (2010). The horseshoe estimator for sparse signals. *Biometrika* 97(2), 465–480. doi:10.1093/biomet/asq017

**Bayesian inference for partially identified parameters**

- Poirier, D. J. (1998). Revising beliefs in nonidentified models. *Econometric Theory* 14(4), 483–509. doi:10.1017/S0266466698144043
- Manski, C. F. (2003). *Partial Identification of Probability Distributions.* Springer. doi:10.1007/b97478
- Gustafson, P. (2005). On model expansion, model contraction, identifiability and prior information: Two illustrative scenarios involving mismeasured variables. *Statist. Sci.* 20(2), 111–140. doi:10.1214/088342305000000098
- Moon, H. R.; Schorfheide, F. (2012). Bayesian and frequentist inference in partially identified models. *Econometrica* 80(2), 755–782. doi:10.3982/ECTA8360
- Gustafson, P. (2015). *Bayesian Inference for Partially Identified Models: Exploring the Limits of Limited Data.* Chapman & Hall/CRC. doi:10.1201/b18308
- Giacomini, R.; Kitagawa, T. (2021). Robust Bayesian inference for set-identified models. *Econometrica* 89(4), 1519–1556. doi:10.3982/ECTA16773

**Two citations in `spec.md` §4 point at the wrong papers.** `doi:10.1016/j.aca.2020.02.048`
resolves to Nagy, Kecskemeti & Gaspar (2020) on immobilised enzyme reactors, not de Juan &
Tauler; the intended paper is the 2021 review above (doi:10.1016/j.aca.2020.10.051).
`doi:10.1016/j.aca.2025.343897` resolves to Tan et al. (2025) on carbon biradical
nanoparticles; the Olivieri 2025 rotational-ambiguity paper is doi:10.1016/j.aca.2025.344304.
Not edited here; the spec is the user's document.
