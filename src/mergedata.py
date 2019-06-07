from __future__ import division
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

with open('data/FinalTable_Nikki(0-10000).pkl', 'rb') as f:
        x = pickle.load(f)

with open('data/FinalTable_10-15-Zoe.pkl', 'rb') as g:
    y = pickle.load(g)

#print(x['flux_list'].tolist())

flux = x['flux_list'].tolist()
wl = x['wavelength'].tolist()

plt.plot(wl[0], flux[0])
plt.show()