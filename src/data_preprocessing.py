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


def plot_one_spectrum(spectra, nth_element, sigma, downsize, filename):
  z = spectra.get_values()[:, 3]
  fluxes = spectra.get_values()[:, 0]
  wavelengths = spectra.get_values()[:, 1]
  gaussian_sigma = sigma
  spectrum_x = spectra.iloc[nth_element]['wavelength']
  spectrum_y = spectra.iloc[nth_element]['flux_list']
  spectrum_title = 'Spectrum with guassian smoothing, sigma = ' + str(gaussian_sigma)
  filename = filename + str(gaussian_sigma)

  spectrum_y_filtered = apply_gaussian_filter(spectrum_y, sigma=gaussian_sigma)
  spectrum_y_downsized = spectrum_y_filtered[::downsize]
  spectrum_x = spectrum_x[::downsize]
  print('len(spectrum_y)', len(spectrum_y))

  fig, ax = plotify.plot(
      x_list=spectrum_x,
      y_list=spectrum_y_downsized,
      xlabel='Frequencies (Hz)',
      ylabel='Flux',
      title=spectrum_title,
      figsize=(12, 8),
      show_plot=True,
      filename=filename,
      ymin=np.amin(spectrum_y) - 4,
      ymax=np.amax(spectrum_y) + 4
  )


# plot_one_spectrum(spectra=spectra, nth_element=434, sigma=2, downsize=4, filename='helloka')


def process_all_objects(spectra=spectra, sigma=16, downsize=8):
  min_wavelength_values = []
  max_wavelength_values = []

  for index, spectrum in spectra.iterrows():
    min_wavelength_values.append(np.amin(spectrum['wavelength'].tolist()))
    max_wavelength_values.append(np.amax(spectrum['wavelength'].tolist()))

  absolute_min = np.amax(np.array(min_wavelength_values))
  absolute_max = np.amin(np.array(max_wavelength_values))

  print('absolute_min', absolute_min)
  print('absolute_max', absolute_max)
  print('min_wavelength_values', min_wavelength_values)

  plotify.plot(
    y_list=min_wavelength_values,
    x_list=range(len(min_wavelength_values)),
    title='Minimum Wavelength Values',
    ymin=3400,
    ymax=6500,
    xlabel='',
    ylabel='Wavelength',
    filename='minimum-wavelength-values'
  )

  plotify.plot(
    y_list=max_wavelength_values,
    x_list=range(len(max_wavelength_values)),
    title='Maximum Wavelength Values',
    ymin = 6000,
    ymax = 14000,
    xlabel='',
    ylabel='Wavelength',
    filename='maximum-wavelength-values'
  )

process_all_objects(spectra=spectra, sigma=8, downsize=4)
