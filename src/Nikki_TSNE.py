import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


spectra = pd.read_pickle('../data/sdss/FinalTable_Nikki.pkl')

print(spectra.iloc[0])

flux_list = spectra.get_values()[:,0]
wavelength = spectra.get_values()[:,1]
objid = spectra.get_values()[:,2]
specclass = spectra.get_values()[:,4]
z = spectra.get_values()[:,7]
z_err = spectra.get_values()[:,5]
dec = spectra.get_values()[:,6]
ra = spectra.get_values()[:,8]



print(len(flux_list))
print(len(z_err))