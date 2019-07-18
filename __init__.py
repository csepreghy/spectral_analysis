import pandas as pd
import time as time

from src.import_data import get_save_SDSS_from_coordinates
from src.SDSS_direct_query import query
from src.merge_tables import merge

from src.data_preprocessing import filter_sources, spectrum_cutoff, create_continuum, merge_lines_and_continuum
from src.get_spectrallines import get_spectrallines
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from matplotlib import style

style.use('fivethirtyeight')

from src.plotify import Plotify

plotify = Plotify()
# with open('data/sdss_coordinates_lowz.txt') as text_file:
#   coord_list = text_file.read().splitlines()
#     mystring.replace('\n', ' ').replace('\r', '')

# query()

# coord_list=pd.read_csv("data/lowz.csv")
# start=time.time()
# ra_list = coord_list["ra"].tolist()
# dec_list= coord_list["dec"].tolist()
# end=time.time()
# tt=end - start
# print("time for listing is:", tt)

# start1=time.time()
# ra=ra_list[40001:45000]
# dec=dec_list[40001:45000]
# get_save_SDSS_from_coordinates( ra , dec )
# end1=time.time()

# tt1 = end1- start1
# length=len(ra) - 1
# print("time for "+str(length)+" stellar objects:" , tt1)

# merge(length)

# start = time.time()

# spectra = pd.read_pickle('data/sdss/FinalTable_Andrew(40-45).pkl')
# df_filtered = filter_sources(df = spectra)
# print('DF Filtered: ')
# print(df_filtered.head())
# df_spectral_lines = get_spectrallines(df_filtered)
# print('Spectral Lines')
# print(df_spectral_lines.head())
# df_spectral_lines.to_pickle('spectral_lines_df.pkl')
# df_cutoff = spectrum_cutoff(df = df_filtered)
# df_continuum = create_continuum(df = df_cutoff, sigma=8, downsize=8)
# df_continuum.to_pickle('continuum_df.pkl')

# df_complete = merge_lines_and_continuum(df_spectral_lines, df_continuum)
# df_complete.to_pickle('COMPLETE_df.pkl')

# end = time.time()
# tt = end - start
# print(" ")
# print("Time elapsed: ", tt, "s")
# print(tt/60, "min")

# model = create_model(df_final, configs['neural_network'])

#df_continuum = pd.read_pickle('continuum_df.pkl')
df_spectral_lines = pd.read_pickle('spectral_lines_df.pkl')

spectra = pd.read_pickle('data/alldatamerged.pkl')
df_filtered = filter_sources(df=spectra)
print('DF Filtered: ')
print(df_filtered.head())
"""
df_spectral_lines = get_spectrallines(df_filtered)
print('Spectral Lines')
print(df_spectral_lines.head())
df_spectral_lines.to_pickle('spectral_lines2_df.pkl')
"""
df_cutoff = spectrum_cutoff(df=df_filtered)
df_continuum = create_continuum(df=df_cutoff, sigma=16, downsize=8)
df_continuum.to_pickle('continuum_df.pkl')


df_complete = merge_lines_and_continuum(df_spectral_lines, df_continuum)
print("DF Complete: ")
print(df_complete.head())
df_complete.to_pickle('COMPLETE_df.pkl')

