import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

# The spectral lines
qso = ['Lya_em', 'C4_ab', 'C4_ab', 'Lyb_ab']
gal = ['OII_em', 'OIII_em', 'OIII_em', 'Na_ab', 'Mg_ab']
star = ['HeI_ab', 'FeII_ab', 'FeII_ab', 'FeII_ab', 'CAHK_ab', 'CAHK_ab']

wl_qso = [1215, 1548, 1551, 1026]
wl_gal = [3727, 4959, 5007, 5892, 5175]
wl_star = [6678, 2600, 2383, 2374, 3934, 3969]

# Load the data and extract the important columns
spectra = pd.read_pickle('../data/sdss/speclines_test.pkl')

print(len(spectra))

print(spectra['spectral_lines'][0])

print(spectra.head())


sys.exit()

flux_list = spectra.get_values()[:,0]
wavelength = spectra.get_values()[:,1]
objid = spectra.get_values()[:,2]
specclass = spectra.get_values()[:,4]
z = spectra.get_values()[:,7]
z_err = spectra.get_values()[:,5]
dec = spectra.get_values()[:,6]
ra = spectra.get_values()[:,8]
#print(spectra.iloc[0])



# Plot the spectra
print(wavelength[0])
#print(len(z_err))

plt.figure(0)
plt.plot(wavelength[0], flux_list[0], lw=0.1)

plt.show()