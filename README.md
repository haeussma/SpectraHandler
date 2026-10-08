# SpectraHandler

Spectral deconvolution for reaction data, built on [JAX](https://docs.jax.dev). From a
time course of absorbance spectra it resolves species concentrations and pure spectra,
fits first-order kinetics per run and pools replicates into rate constants with an
interval. Without a kinetic model, it reports the band of every split the data allow,
rather than one answer.

## Install

```bash
uv add spectrahandler            # or: pip install spectrahandler
uv add "spectrahandler[plot]"    # with matplotlib, for spectrahandler.plot
```

Python 3.13 or newer.

## Use

```python
import jax

jax.config.update("jax_enable_x64", True)  # first, before anything creates arrays

from spectrahandler.curve_resolution import SpectralDataset
from spectrahandler.kinetic_studio import read_kinetic_studio
from spectrahandler.kinetics import Scheme, fit_kinetics
from spectrahandler.plot import plot_deconvolution, plot_residuals

scans = [read_kinetic_studio(f"shot{i}.csv") for i in (1, 2, 3)]
data = SpectralDataset.from_runs(
    [(scan.time_s, scan.absorbance) for scan in scans],
    wavelength=scans[0].wavelength_nm,
    species=("A", "B"),
    initial_state={"A": 7.0},  # concentrations at the first time label
    run_ids=["shot1", "shot2", "shot3"],
    conditions=["250uM"] * 3,  # same label = replicates
    time_unit="s",  # the default is hours
)

fit = fit_kinetics(data, Scheme(steps=[("A", "B")]))
summary = fit.summary()["250uM"]
rate = summary.rates[("A", "B")]
print(f"k = {rate.value:.3f} 1/s, 95 % interval [{rate.lower:.3f}, {rate.upper:.3f}]")
print(summary.warnings)

plot_deconvolution(fit, data, "shot1")  # concentrations and spectra of one run
plot_residuals(fit, data, "shot1")  # what the fit leaves, in noise units
```

JASCO exports are read with `spectrahandler.jasco`. The guides in [`docs/`](docs/) cover
curve resolution, kinetic fits and plotting step by step.

## Sharp bits

- **float64 is yours to enable.** The library never turns it on; `fit_kinetics` and
  `resolve_band` refuse to run in float32.
- **Units are labels you set.** `time_unit` defaults to `"h"`. Rates come out per
  `time_unit`, concentrations in the unit of `initial_state`.
- **Species are sorted alphabetically.** Every species axis follows `data.species`, not
  the order you passed.
- **Arrays are `(run, time, wavelength)`.** A single run is `n_run == 1`. Runs of
  different lengths go through `from_runs`, which pads and masks them.
- **No missing values where measured.** Readers keep empty cells as `NaN`, and
  `SpectralDataset` rejects them. Crop the wavelength range first; nothing is dropped
  for you.
- **The first time label is time zero.** A dead time or a time offset does not move the
  rates, but the spectra of the starting species and of any intermediate absorb it. Pass
  `t_offset` to `fit_kinetics` if you know it.
- **The interval covers only your declared replicates.** Shots from one loading say
  nothing about preparation-to-preparation or day-to-day variation. One run gives no
  interval at all.
- **Read `summary.warnings`.** They flag non-converged runs, replicate spread no larger
  than the fit precision, and rates that can be swapped with an equally good fit
  (A → B → C with a free intermediate spectrum).
- **The band is not a posterior.** `resolve_band` needs fully measured runs and returns
  every split the data allow; its spectra have a run axis of size 1, shared by all runs.
- **Plots take a run id, not an index.** The fit and the dataset must have the same
  `run_ids`.
- **Kinetic Studio filenames lie.** The `_NNNs` in a filename is the acquisition window,
  not a reaction time.

## Develop

```bash
uv sync
pre-commit install
uv run pytest
cd docs && npm install && npm run dev   # documentation site
```
