import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from skimage import io, filters, feature
from scipy import ndimage

from plotify import Plotify

plotify = Plotify()
spectra = pd.read_pickle('data/sdss/FinalTable.pkl')

def apply_gaussian_filter(fluxes, sigma):
  return filters.gaussian(image=fluxes, sigma=sigma)


z = spectra.get_values()[:,3]
fluxes = spectra.get_values()[:,0]
wavelengths = spectra.get_values()[:, 1]
gaussian_sigma = 64

spectrum_x = spectra.iloc[42]['wavelength']
spectrum_y = spectra.iloc[42]['flux_list']
print('spectra.iloc[42][ra]', type(spectra.iloc[42]['ra']))
spectrum_title = 'Spectrum with guassian smoothing, sigma = ' + str(gaussian_sigma)
filename = 'gaussian-' + str(gaussian_sigma)

spectrum_y = apply_gaussian_filter(spectrum_y, sigma=gaussian_sigma)
spectrum_y = spectrum_y[::8]
spectrum_x = spectrum_x[::8]
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
