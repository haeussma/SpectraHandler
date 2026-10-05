"""Bayesian curve resolution: spectra and concentration profiles, with posteriors."""

from spectrahandler.curve_resolution.band import FeasibleBand, resolve_band
from spectrahandler.curve_resolution.dataset import SpectralDataset
from spectrahandler.curve_resolution.inference import fit
from spectrahandler.curve_resolution.model import curve_resolution_model
from spectrahandler.curve_resolution.synthetic import make_easy_dataset, make_realistic_dataset

__all__ = [
    "FeasibleBand",
    "SpectralDataset",
    "curve_resolution_model",
    "fit",
    "make_easy_dataset",
    "make_realistic_dataset",
    "resolve_band",
]
