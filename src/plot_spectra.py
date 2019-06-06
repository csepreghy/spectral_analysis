import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from skimage import io, filters, feature
from scipy import ndimage

from plotify import Plotify

plotify = Plotify()
spectra = pd.read_pickle('../data/sdss/FinalTable_Nikki.pkl')


print(spectra.iloc[0])

z = spectra.get_values()[:,3]
fluxes = spectra.get_values()[:,0]
wavelengths = spectra.get_values()[:, 1]

spectrum_x = spectra.iloc[1]['wavelength']
spectrum_y = spectra.iloc[1]['flux_list']
print('spectra.iloc[0][ra]', type(spectra.iloc[0]['ra']))
spectrum_title = 'Spectrum for coordinates: ' + str(spectra.iloc[0]['ra']) + ', ' +  str(spectra.iloc[0]['dec'])


print('spectrum_x', spectrum_x)
print('spectrum_y', spectrum_y)

# plotify.plot(
#   x_list = spectrum_x, 
#   y_list = spectrum_y,
#   xlabel = 'Frequencies (Hz)',
#   ylabel = 'Flux'
# )

def apply_gaussian_filter(fluxes, sigma = 2):
  return filters.gaussian(image=fluxes, sigma=sigma)

spectrum_y = apply_gaussian_filter(spectrum_y, sigma=20)


fig, ax = plotify.plot(
  x_list = spectrum_x,
  y_list = spectrum_y,
  xlabel = 'Frequencies (Hz)',
  ylabel = 'Flux',
  title = spectrum_title,
  figsize=(12, 8)
)

