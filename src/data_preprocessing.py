import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from skimage import io, filters, feature
from scipy import ndimage
import pickle
from itertools import islice

from plotify import Plotify

# --- Initialize variables --- #
plotify = Plotify()
spectra = pd.read_pickle('data/sdss/FinalTable.pkl')

CUTOFF_MIN = 3850
CUTOFF_MAX = 9100

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

# FUNCTION DESCROPTION:
#
# name: filter_sources
# input: (df) all sources with columns = [class, dec, flux_list, objid, plate, ra, wavelength, z, zErr]
# output: (df) filtered df that removes all sources that fall out of the optimal wavelength range

def filter_sources(df):
  rows_after_removal = []

  print('Number of rows before filtering: ', str(len(df)))

  for index, spectrum in df.iterrows():
    min_value = np.amin(spectrum['wavelength'].tolist())
    max_value = np.amax(spectrum['wavelength'].tolist())

    if min_value < CUTOFF_MIN and max_value > CUTOFF_MAX:
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

      rows_after_removal.append(row)
    
  filtered_df = pd.DataFrame(rows_after_removal)
  print('Number of rows after filtering: ', str(len(filtered_df)))

  return filtered_df

# filtered_df = filter_sources(df=spectra)
# filtered_df.to_pickle('filtered_df.pkl')

with open('filtered_df.pkl', 'rb') as f:
  filtered_df = pickle.load(f)

def spectrum_cutoff(df):
  rows_after_cutoff = []
  for df_index, spectrum in df.iterrows():
    cut_off_wavelengths = []
    cut_off_fluxes = []

    wavelengths = spectrum['wavelength']
    fluxes = spectrum['flux_list']
  

    for wavelength, flux in zip(wavelengths, fluxes):
      if wavelength > CUTOFF_MIN and wavelength < CUTOFF_MAX:
        cut_off_wavelengths.append(wavelength)
        cut_off_fluxes.append(flux)

    row = {
      'wavelength': cut_off_wavelengths,
      'flux_list': cut_off_fluxes,
      'objid': spectrum['objid'],
      'plate': spectrum['plate'],
      'class': spectrum['class'],
      'zErr': spectrum['zErr'],
      'dec': spectrum['dec'],
      'ra': spectrum['ra'],
      'z': spectrum['z'],
    }

    rows_after_cutoff.append(row)
  
  filtered_df = pd.DataFrame(rows_after_cutoff)

  return filtered_df

df_after_cutoff = spectrum_cutoff(filtered_df)
print('df_after_cutoff', df_after_cutoff)

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

