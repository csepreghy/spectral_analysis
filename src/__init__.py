import pandas as pd
import time as time
import numpy as np
import matplotlib.pyplot as plt

from data_preprocessing import filter_sources, spectrum_cutoff, create_continuum, merge_lines_and_continuum
from get_spectrallines import get_spectrallines

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_blobs

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

from plotify import Plotify

# 1) Get coordinates from query - #
# 2) Import data - # 
# 3) Merge spectra with table containing other information - #
# 4) Filter Out Spectra with not enough values - #
# 5) Merge all data - #
# 6) Get spectral lines - #
# 7) Cut off values from the sides to have the same range for all spectra - #
# 8) Merge spectral lines with the continuum to one table
# 9) Run the ML algorithms - #



df_spectra = pd.read_pickle('data/spectra-meta-0-70k.pkl')


# df_filtered = filter_sources(df = df_spectra)
# df_filtered.to_pickle('data/spectra-meta-filtered_0-70k.pkl')
# df_filtered.to_msgpack('data/spectra-meta-filtered_0-70k.msg')
df_filtered = pd.read_msgpack('data/spectra-meta-filtered_0-70k.msg')

print('DF Filtered: ')
print(df_filtered.columns)
print(df_filtered.head())

df_spectral_lines = get_spectrallines(df_filtered)
print('Spectral Lines')
print(df_spectral_lines.head())

df_spectral_lines.to_pickle('data/spectral_lines_df_0-70k.pkl')
# # df_spectral_lines = pd.read_pickle('spectral_lines_df_5001-10000.pkl')

df_cutoff = spectrum_cutoff(df = df_filtered)
print('DF Cutoff: ')
print(df_cutoff.columns)
print(df_cutoff)

df_continuum = create_continuum(df = df_cutoff, sigma=8, downsize=8)
df_continuum.to_pickle('data/continuum_df_0-70k.pkl')
# df_continuum = pd.read_pickle('data/continuum_df_5001-10000.pkl')
print('DF Continuun: ')
print(df_continuum.columns)
print(df_continuum)

df_preprocessed = merge_lines_and_continuum(df_spectral_lines, df_continuum)
df_preprocessed.to_pickle('data/preprocessed_0-70k.pkl')
# df_preprocessed = pd.read_pickle('data/preprocessed_5001-10000.pkl')

print('DF Preprocessed (Final)')
print(df_preprocessed.columns)
print(df_preprocessed.head())

# model = create_model(df_preprocessed, configs['neural_network'])

# df_continuum = pd.read_pickle('continuum_df.pkl')
# df_spectral_lines = pd.read_pickle('spectral_lines_df.pkl')

# spectra = pd.read_pickle('data/alldatamerged.pkl')
# df_filtered = filter_sources(df=spectra)
# print('DF Filtered: ')
# print(df_filtered.head())

# """
# df_spectral_lines = get_spectrallines(df_filtered)
# print('Spectral Lines')
# print(df_spectral_lines.head())
# df_spectral_lines.to_pickle('spectral_lines2_df.pkl')
# """
# df_cutoff = spectrum_cutoff(df=df_filtered)
# df_continuum = create_continuum(df=df_cutoff, sigma=16, downsize=8)
# df_continuum.to_pickle('continuum_df.pkl')


# df_complete = merge_lines_and_continuum(df_spectral_lines, df_continuum)
# print("DF Complete: ")
# print(df_complete.head())
# df_complete.to_pickle('COMPLETE_df.pkl')

