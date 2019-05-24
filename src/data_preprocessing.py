import numpy as np
from astropy.io import fits

fits_image_filename = fits.util.get_testdata_filepath('http://www.gama-survey.org/dr3/data/spectra/gama/reduced_27/1d/G02_Y3_001_004.fit')
hdul = fits.open(fits_image_filename)
print(hdul.info())
hdul.close()
