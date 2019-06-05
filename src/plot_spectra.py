import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys


spectra = pd.read_pickle('data/sdss/60001-65000.pkl')


print(spectra.iloc[0])

z = spectra.get_values()[:,3]
fluxes = spectra.get_values()[:,0]


print((fluxes[0]))
print(len(fluxes))
