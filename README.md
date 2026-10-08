# SpectraHandler

Spectral deconvolution for reaction data, built on [JAX](https://docs.jax.dev). From a
time course of absorbance spectra it resolves species concentrations and pure spectra,
fits first-order kinetics per run and pools replicates into rate constants with an
interval. Without a kinetic model, it reports the band of every split the data allow,
rather than one answer.

## Install

```bash
uv add git+https://github.com/haeussma/SpectraHandler
# or: pip install git+https://github.com/haeussma/SpectraHandler
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

## Develop

```bash
uv sync
pre-commit install
uv run pytest
cd docs && npm install && npm run dev   # documentation site
```
