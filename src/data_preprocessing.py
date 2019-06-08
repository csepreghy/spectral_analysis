import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from skimage import io, filters, feature
from scipy import ndimage
from itertools import islice

from plotify import Plotify

# --- Initialize variables --- #
plotify = Plotify()
spectra = pd.read_pickle('data/sdss/FinalTable.pkl')


def apply_gaussian_filter(fluxes, sigma):
  return filters.gaussian(image=fluxes, sigma=sigma)


def plot_one_spectrum(spectra, nth_element, sigma, downsize, filename, save):
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
    ymax=np.amax(spectrum_y) + 4,
    save=save
  )


# plot_one_spectrum(spectra=spectra, nth_element=41, sigma=4, downsize=1, filename='helloka', save=False)

# 3845 is the max

def cutoff(spectra=spectra, sigma=16, downsize=8):
  min_wavelength_values = []
  max_wavelength_values = []

  df = spectra

  # for index, spectrum in islice(spectra.iterrows(), 5000):
  #   min_value = np.amin(spectrum['wavelength'].tolist())
  #   max_value = np.amax(spectrum['wavelength'].tolist())

  #   if min_value < 3850: min_wavelength_values.append(min_value)
  #   if max_value > 9100: max_wavelength_values.append(max_value)

  # absolute_min = np.amax(np.array(min_wavelength_values))
  # absolute_max = np.amin(np.array(max_wavelength_values))


  rows_after_cutoff = []

  for index, spectrum in spectra.iterrows():
    min_value = np.amin(spectrum['wavelength'].tolist())
    max_value = np.amax(spectrum['wavelength'].tolist())

    if min_value < 3553 and max_value > 9100:
      row = {
        'wavelength': spectrum['wavelength'].tolist(),
        'flux_list': spectrum['flux_list'].tolist(),
        'objid': spectrum['objid'],
        'plate': spectrum['plate'],
        'class': spectrum['class'],
        'zErr': spectrum['zErr'],
        'dec': spectrum['dec'],
        'ra': spectrum['ra'],
        'z': spectrum['z'],
      }

      rows_after_cutoff.append(row)
    
  cut_df = pd.DataFrame(rows_after_cutoff)


  # df.loc[(np.amin(df['wavelength'].tolist()) < 3850) & 
  #        (np.amax(df['wavelength'].tolist()) < 9100)]




cutoff(spectra=spectra)


def check_minmax_values(spectra=spectra, sigma=16, downsize=8):
  min_wavelength_values = []
  max_wavelength_values = []

  for index, spectrum in islice(spectra.iterrows(), 500):
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
    ymin=6000,
    ymax=14000,
    xlabel='',
    ylabel='Wavelength',
    filename='maximum-wavelength-values'
  )

