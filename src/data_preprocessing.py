import numpy as np
from astropy.io import fits
import urllib.request
import pandas as pd

url = 'http://www.gama-survey.org/dr3/data/spectra/gama/reduced_27/1d/G02_Y3_001_004.fit'
urllib.request.urlretrieve(url, 'G02_Y3_001_004.fit')

# fits_image_filename = fits.util.get_testdata_filepath('G02_Y3_001_004.fit')
hdul = fits.open('G02_Y3_001_004.fit')

with fits.open('G02_Y3_001_004.fit') as data:
  df = pd.DataFrame(data[0].data)

  for index, item in enumerate(data[0].header):
    print(item, data[0].header[index])

print(df.head())
