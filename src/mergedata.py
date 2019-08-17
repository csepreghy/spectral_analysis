from __future__ import division
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

def merge_data(filenames):
  with open('data/sdss/spectra-meta/' + filenames[0], 'rb') as g:
    data = pickle.load(g)

  for i in range(1,len(filenames)):
    with open('data/sdss/spectra-meta/' + filenames[i], 'rb') as f:
      x = pickle.load(f)
      datalist = pd.concat([data, x], ignore_index=True)
      data = datalist

  return data

path = 'data/sdss/spectra-meta'
filenames = [f for f in listdir(path) if isfile(join(path, f))]

all_data = merge_data(filenames)
all_data.to_pickle('data/alldatamerged_test.pkl')

df = pd.read_pickle('data/alldatamerged_test.pkl')
print('df', df)