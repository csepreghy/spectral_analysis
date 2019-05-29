import numpy as np
from astropy.io import fits

fits_image_filename = fits.util.get_testdata_filepath('http://www.gama-survey.org/dr3/data/spectra/gama/reduced_27/1d/G02_Y3_001_004.fit')
hdul = fits.open(fits_image_filename)
print(hdul.info())
hdul.close()

#GAMA
wavelength = []
increase = 5129/4954
for i in range (0,4954):
    wavelength.append(3727+i*increase)

#plot GAMA
#plt.errorbar(wavelength, list(data)[0], list(data)[1], marker='.', linestyle="", color='black', alpha=0.5, ms=1.5)

#SDSS plotting
fig, ax = plt.subplots(figsize=(14.,8.5))
for i in np.arange(xid['ra'].size):
    ax.plot(10.**sp[i][1].data['loglam'],sp[i][1].data['flux'],label=xid['instrument'][i])
ax.set_ylabel('Flux [10$^{-17}$ ergs/cm$^2$/s/\AA]')
ax.set_xlabel('Wavelength [\AA]')