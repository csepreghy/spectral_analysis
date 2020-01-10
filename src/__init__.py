import pandas as pd
import time as time
import numpy as np
import matplotlib.pyplot as plt

from src.downloading import download_spectra
from data_preprocessing import filter_sources, spectrum_cutoff, create_continuum, merge_lines_and_continuum
from src.merge_tables import merge

from get_spectrallines import get_spectrallines

# 1) Get coordinates from query - #
# 2) Download data - # 
# 3) Merge spectra with table containing other information - #
# 4) Filter Out Spectra with not enough values - #
# 5) Merge all data - #
# 6) Get spectral lines - #
# 7) Cut off values from the sides to have the same range for all spectra - #
# 8) Merge spectral lines with the continuum to one table
# 9) Run the ML algorithms - #

# ----------------------------------------------- #
# -------------- 2) Download Data --------------- #
# ----------------------------------------------- #

df_raw_specrta = download_spectra(coord_list_url = "data/sdss/coordinate_list.csv",
								  from_sp = 5001,
								  to_sp = 5010,
								  save=False)

df_spectra = pd.read_pickle('data/sdss/spectra-meta/spectra-meta-merged_10001-20000.pkl')
print(f'df_spectra = {df_spectra}')

# df_filtered = filter_sources(df = df_spectra)
# df_filtered.to_pickle('data/spectra-meta-filtered_0-70k.pkl')
# df_filtered.to_msgpack('data/spectra-meta-filtered_0-70k.msg')
# df_filtered = pd.read_msgpack('data/spectra-meta-filtered_0-70k.msg')

# print('DF Filtered: ')
# print(df_filtered.columns)
# print(df_filtered.head())

# df_spectral_lines = get_spectrallines(df_filtered)
# print('Spectral Lines')
# print(df_spectral_lines.head())

# df_spectral_lines.to_pickle('data/spectral_lines_df_0-70k.pkl')
# #s df_spectral_lines = pd.read_pickle('spectral_lines_df_5001-10000.pkl')

# df_cutoff = spectrum_cutoff(df = df_filtered)
# print('DF Cutoff: ')
# print(df_cutoff.columns)
# print(df_cutoff)

def get_df_continuum():
	"""
	get_df_continuum()

	A wrapper for craete_continuum in order to better structure the __init__.py
	
	Makes Use Of
	------------
	create_continuum() --imported from src/data_preprocessing.py
	
	Returns
	-------
	out: DataFrame()
		A pandas dataframe that has the flux_list which went through Gaussian smoothing
		and a downsizing with the factor tunable here.
	"""
	df_continuum = create_continuum(df = df_cutoff, sigma=8, downsize=8)
	df_continuum.to_pickle('data/continuum_df_0-70k.pkl')
	# df_continuum = pd.read_pickle('data/continuum_df_5001-10000.pkl')
	print('DF Continuun: ')
	print(df_continuum.columns)
	print(df_continuum)

# df_preprocessed = merge_lines_and_continuum(df_spectral_lines, df_continuum)
# df_preprocessed.to_pickle('data/preprocessed_0-70k.pkl')
# # df_preprocessed = pd.read_pickle('data/preprocessed_5001-10000.pkl')

# print('DF Preprocessed (Final)')
# print(df_preprocessed.columns)
# print(df_preprocessed.head())

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


def main():
	get_df_continuum()


if __name__ == '__main__':
	main()
