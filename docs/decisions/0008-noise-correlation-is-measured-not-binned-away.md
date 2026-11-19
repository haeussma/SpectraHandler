---
status: accepted
date: 2026-11-19
implements: [src/spectrahandler/curve_resolution/noise.py, src/spectrahandler/curve_resolution/band.py]
---

# 0008 — Noise shared between channels is measured and modelled, not binned away

## Context

[ADR 0007](0007-credible-band-posterior-for-the-identified-part.md) takes the noise as
independent between all points, with one level per wavelength. Real detectors break both
halves of that. The remedy so far was to bin channels by hand, with a guessed width.

**What the blanks of EXP0001 show** (raw export, 787 channels of 0.46 nm, 297 noise spectra
from frame-to-frame differences of three buffer blanks; the data stay local):

- Along wavelength the correlation falls in a straight line, 0.92, 0.83, 0.74, … to 0.05 at
  lag 11, and stays at 0.05 out to lag 40. The partial autocorrelation is 0.92 at lag 1,
  about zero for lags 2–11, and 0.44 at lag 12. That is a moving average over 12 channels
  plus a part that the whole spectrum shares. It is **not AR(1)**, which would give 0.42 at
  lag 10 where 0.13 is measured.
- The same in every wavelength region (lag-1 0.914–0.919), so it is a property of the
  readout, not of the sample.
- Along time the noise is white (lag-1 of the differenced series −0.48; white gives −0.5).
- Binning by 10 leaves about 0.25 correlation between neighbouring bins plus the shared
  part, so it does not make the noise independent either.
- Whether the noise grows with absorbance cannot be read from the blanks: the enzyme blank
  is only 0.1 AU above buffer.

Photon noise is independent per pixel, so the averaging happens in the electronics or the
software, before export.

## Decision

1. **The correlation between channels is measured from the data, with no instrument
   parameter.** `measure_noise_correlation` takes any static sample (a blank, or a stretch
   of a run in which the spectra no longer change), forms the differences of consecutive
   spectra, scales every channel by its own noise level, and averages the power spectra.
   The result, `NoiseCorrelation`, holds the power spectrum along the channel index and the
   fraction shared by the whole spectrum. It carries over to any grid with the same channel
   spacing.
2. **The fit uses it through the Whittle likelihood.** The residual spectra, scaled by the
   noise level, are taken to have that power spectrum on a circular grid. This costs one
   FFT per spectrum instead of a dense covariance.
3. **The noise level stays a posterior**, one per wavelength. The correlation is a measured
   input: 100 blank spectra give every frequency to 10 %, far more precisely than the fit
   could.
4. **Growth with absorbance is one exponent**: the noise standard deviation is multiplied
   by `10^(γ·(A − Ā))`, with `A` the modelled absorbance and `Ā` its mean at that
   wavelength. `noise_exponent` fixes γ (default 0; 0.5 is photon counting) or, as `None`,
   samples it with a normal prior of mean 0.5 and sd 0.5.
5. **The sampler works in noise-whitened coordinates.** The gauge of ADR 0007 is set up
   after whitening along wavelength, so the latents are equally well determined.
6. **The defaults are unchanged**: without `noise_correlation` the channels are independent.

### What was tried and dropped

- **Estimating the correlation from the reaction data** by removing the strongest patterns
  of the differences first. The removed patterns are smooth, as the slow part of the noise
  is, so the estimate came out too weakly correlated (−0.07 at lag 11 where 0.13 was true)
  and the noise level 5–10 % too low. That errs towards overconfidence, so only static
  samples are accepted.
- **Building the power spectrum from a truncated autocorrelation.** Its error at high
  frequency was as large as the power there, and the fit over-trusted fine detail.
  Averaged power spectra have the same relative error at every frequency.
- **A parametric shape** (moving-average width, AR coefficient) sampled in the model. It
  would tie the library to one kind of detector, and the measurement is already precise.

## Consequences

**Synthetic data** (256 channels, noise averaged over 12 channels, 5 % shared, 5 % white,
six noise seeds, one reference):

| | independent noise assumed | measured correlation |
|---|---|---|
| true amounts of the referenced species inside the band | 0.90–1.00, below 0.95 in three of six | 1.00 in all |
| true spectrum values inside the band | 0.96–0.98 | 0.96–1.00 |
| median width of the amount band | 1.3–1.6 µM | 2.0–2.2 µM |
| time | 13 s | 40 s |

On a second set (128 channels, average over 8) the independent model finds no feasible
split at all: with the wrong noise model, true zeros look significantly negative.

The exponent is recovered as 0.52 ± 0.06 (true 0.5) and 0.03 ± 0.07 (true 0) when the
spectra reach 1.3 AU, and as 0.57 ± 0.23 when they reach only 0.33 AU.

**EXP0001, all 787 channels, three 1 s shots**, correlation from the buffer blank, exponent
sampled, backgrounds as in ADR 0004 with the tolerance calibrated on the blank:

| | binned by 10, independent | unbinned, independent | unbinned, measured correlation and exponent |
|---|---|---|---|
| noise exponent | fixed 0 | fixed 0 | **0.54 ± 0.03** |
| median noise level (buffer blank: 0.00067 AU) | 0.00084 (per bin) | 0.00094 | 0.00070 |
| converted at 1 s (µM) | 4.9–6.9 | 5.0–6.8 | 5.0–7.0 |
| ε364 of the intermediate (mM⁻¹ cm⁻¹; Fraaije 1997: 46) | 43.7–57.1 | 44.3–57.8 | 43.5–57.0 |
| ε439 of the oxidised enzyme (12.5) | 11.9–13.6 | 11.8–13.7 | 11.8–13.7 |
| split R-hat | 1.005 | 1.009 | 1.029 |
| time | 10 s | 116 s | 649 s |

- **The data confirm photon-counting noise.** The exponent comes out at the shot-noise
  value, and with it the inferred noise level matches the blank. Without it the model
  reported 40 % more noise than the blank shows.
- **The bands hardly move on this data set**, because their width is set by the split and
  by the baseline tolerance, not by noise. The gain is that the noise posterior is now
  right and no bin width was chosen.

**Limits.**

- The correlation is assumed to depend only on the channel distance and to be the same
  along the whole grid. Channels must be equally spaced.
- The Whittle likelihood treats the grid as circular, which is accurate only when the
  correlation reaches over far fewer channels than the grid has.
- A static sample is needed. A reaction in progress distorts the measurement.
- The correlation's own uncertainty is not propagated.
- Unbinned fits are slow: minutes for hundreds of channels and spectra.
- Correlation in time is not modelled. The per-shot static pattern of
  [ADR 0005](0005-kinetic-fit-per-replicate-uncertainty-from-replicates.md) is a different
  thing and is still not part of the noise model.
