"""Bayesian curve resolution: spectra and concentration profiles, with posteriors."""

from spectrahandler.curve_resolution.dataset import SpectralDataset
from spectrahandler.curve_resolution.inference import fit
from spectrahandler.curve_resolution.model import curve_resolution_model
from spectrahandler.curve_resolution.synthetic import make_easy_dataset

__all__ = ["SpectralDataset", "curve_resolution_model", "fit", "make_easy_dataset"]
