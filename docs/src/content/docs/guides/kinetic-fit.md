---
title: Rate constants from replicate time courses
description: Fit a first-order reaction scheme to each run, pool replicates by condition, and report rate constants with an interval that comes from the replicates.
sidebar:
  order: 4
---

When you know the reaction scheme, the time course of the spectra pins down both the
rate constants and the spectrum of every species. `fit_kinetics` fits the scheme to each
run on its own; `summary()` pools the replicates of each condition.

## Run it

```python
import jax

jax.config.update("jax_enable_x64", True)  # spectrahandler needs float64

from spectrahandler.kinetics import Scheme, fit_kinetics, make_kinetic_replicates

scheme = Scheme(steps=[("A", "B")])  # A -> B, first order
data, true_log_rates, true_spectra = make_kinetic_replicates(
    jax.random.key(0),
    scheme,
    {("A", "B"): 0.8},  # 0.8 per second
    initial={"A": 10.0},
    n_replicates=4,
    between_sd_log=0.02,
)

fit = fit_kinetics(data, scheme)
rate = fit.summary()["synthetic"].rates[("A", "B")]
print(f"k = {rate.value:.3f} 1/s, 95 % interval [{rate.lower:.3f}, {rate.upper:.3f}]")
print(fit.rates[:, 0])  # one rate per replicate
```

Your own data go in through `SpectralDataset.from_runs`, one `(time, absorbance)` pair per
run. Runs with the same `conditions` label are replicates. Stopped-flow exports from
Kinetic Studio are read with `spectrahandler.kinetic_studio.read_kinetic_studio`.

```python
from spectrahandler.curve_resolution import SpectralDataset
from spectrahandler.kinetic_studio import read_kinetic_studio

paths = ["run1.csv", "run2.csv", "run3.csv"]
scans = [read_kinetic_studio(p) for p in paths]
data = SpectralDataset.from_runs(
    [(scan.time_s, scan.absorbance) for scan in scans],
    wavelength=scans[0].wavelength_nm,
    species=("A", "B", "C"),
    initial_state={"A": 10.0},  # per run: pass a list of dicts
    run_ids=["run1", "run2", "run3"],
    conditions=["condition A"] * 3,
    time_unit="s",
)
```

Schemes are lists of steps: `[("A", "B"), ("B", "C")]` is A → B → C, and
`[("A", "B"), ("B", "A")]` is a reversible pair. For more than one step, pass
`initial_rates` with a rough guess per step, in 1 / `time_unit`:

```python
fit = fit_kinetics(
    data,
    Scheme(steps=[("A", "B"), ("B", "C")]),
    initial_rates={("A", "B"): 1.0, ("B", "C"): 0.1},
)
```

## Where the interval comes from

Each run is fitted by maximum likelihood, with the noise level of every wavelength
estimated in the same fit. The rate's interval does **not** come from that fit: within
one run the fit is far more confident than repeated runs agree, because real noise is
correlated and every run carries small differences of its own. The interval comes from
how much your replicates disagree: the geometric mean of the replicate rates, with a
Student-t interval on n − 1 degrees of freedom.

So:

- **One run gives a rate but no interval.** The summary says so.
- **The interval covers only what differs between your replicates.** Three shots from one
  syringe loading say nothing about the next enzyme preparation.
- **`within_sd_log`** is the single-run precision, as a diagnostic. When replicates do not
  spread clearly beyond it, the summary warns.

## What the fit cannot tell apart

- **A constant offset.** A baseline that does not change in time adds equally to every
  species' spectrum. Rates are unaffected; the spectra include the offset.
- **The true time zero.** Shifting the time axis leaves the rates and the spectra of end
  products unchanged. The spectra of the starting species and of every intermediate absorb
  the shift: they describe the mixture present at your first time label, not at the true
  start of the reaction. If you know your instrument's dead time, pass it as `t_offset`.
- **Swapped rates in a chain.** In A → B → C with an unknown spectrum for B, the two
  rates can be exchanged with an equally good fit. `fit.ambiguities` lists such cases and
  the summary warns; it does not choose for you.

## Check the fit

```python
import matplotlib.pyplot as plt
from spectrahandler.plot import plot_kinetic_fit

plot_kinetic_fit(fit, data, wavelengths=[400, 520])
plt.show()
```

The right panel shows what is left after the fit, in units of each wavelength's noise
level. Even static is good. Stripes along wavelength or blocks along time mean the
scheme, or the assumption of no drifting baseline, does not hold, and the rates are then
the best the chosen scheme can do, not the truth.
