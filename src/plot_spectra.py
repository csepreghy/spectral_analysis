import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys


spectra = pd.read_pickle('/Users/nikki/Documents/Machine Learning/Big project/spectral-analysis/data/sdss_csv/Test.pkl')


print(spectra.iloc[0])

z = spectra.get_values()[:,3]
fluxes = spectra.get_values()[:,0]


print((fluxes[0]))
print(len(fluxes))