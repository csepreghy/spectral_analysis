import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from skimage import io, filters, feature
from scipy import ndimage

from plotify import Plotify

# --- Initialize variables --- #
plotify = Plotify()
spectra = pd.read_pickle('data/sdss/FinalTable.pkl')


def apply_gaussian_filter(fluxes, sigma):
  return filters.gaussian(image=fluxes, sigma=sigma)

def plot_one_spectrum(spectra, nth_element, sigma, downsize):
  z = spectra.get_values()[:,3]
  fluxes = spectra.get_values()[:,0]
  wavelengths = spectra.get_values()[:, 1]
  gaussian_sigma = sigma
  spectrum_x = spectra.iloc[42]['wavelength']
  spectrum_y = spectra.iloc[42]['flux_list']
  spectrum_title = 'Spectrum with guassian smoothing, sigma = ' + str(gaussian_sigma)
  filename = 'gaussian-' + str(gaussian_sigma)

  spectrum_y = apply_gaussian_filter(spectrum_y, sigma=gaussian_sigma)
  spectrum_y = spectrum_y[::downsize]
  spectrum_x = spectrum_x[::downsize]
  print('len(spectrum_y)', len(spectrum_y))
    
  fig, ax = plotify.plot(
    x_list = spectrum_x,
    y_list = spectrum_y,
    xlabel = 'Frequencies (Hz)',
    ylabel = 'Flux',
    title = spectrum_title,
    figsize=(12, 8),
    show_plot=True,
    filename=filename
  )


plot_one_spectrum(spectra=spectra, nth_element=100, sigma=8, downsize=8)


