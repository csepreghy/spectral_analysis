import pandas as pd
import time as time
import numpy as np
import matplotlib.pyplot as plt

from sdss_direct_query import get_coordinates_from_query
from downloading import download_spectra
from data_preprocessing import filter_sources, spectrum_cutoff, create_continuum, merge_lines_and_continuum
from merge_tables import merge_with_metatable

from get_spectrallines import get_spectrallines

# 1) Get coordinates from query ------------------------------------------- #
# 2) Download data -------------------------------------------------------- # 
# 3) Merge spectra with table containing meta information ----------------- #
# 4) Filter Out Spectra with not enough values ---------------------------- #
# 5) Cut off values from the sides to have the same range for all spectra - #
# 6) Create Continuum that has Gaussian smoothing ------------------------- #
# 7) Get spectral lines --------------------------------------------------- #
# 8) Merge spectral lines with the continuum to one table ----------------- #
# 9) Merge all data ------------------------------------------------------- #
# 10) Run the ML algorithms ----------------------------------------------- #


def main():
	from_sp = 1000
	to_sp = 2020

# --------------------------------------------------------------------------- #
# ----------------------- 1) Get cordinates from query ---------------------- #
# --------------------------------------------------------------------------- #

	#get_coordinates_from_query(save_metatable=False, save_coordinates=False)

# --------------------------------------------------------------------------- #
# ----------------------------- 2) Download Data ---------------------------- #
# --------------------------------------------------------------------------- #

	df_raw_spectra = download_spectra(coord_list_url="data/sdss/coordinate_list.csv",
	 								  from_sp=from_sp,
	 								  to_sp=to_sp,
	 								  save=True)

# --------------------------------------------------------------------------- #
# --------- 3) Merge spectra with table containing meta information --------- #
# --------------------------------------------------------------------------- #

	df_merged = merge_with_metatable(from_sp=str(from_sp), to_sp=str(to_sp), save=False)

# --------------------------------------------------------------------------- #
# --------------- 4) Filter Out Spectra with not enough values -------------- #
# --------------------------------------------------------------------------- #

	df_filtered = filter_sources(df=df_merged, save=False)

# --------------------------------------------------------------------------- #
# - 5) Cut off values from the sides to have the same range for all spectra - #
# --------------------------------------------------------------------------- #
	
	df_cutoff = spectrum_cutoff(df=df_filtered)

# --------------------------------------------------------------------------- #
# ------------- 6) Create Continuum that has Gaussian smoothing ------------- #
# --------------------------------------------------------------------------- #

	df_continuum = create_continuum(df=df_cutoff, sigma=8, downsize=8, save=False)
	
# --------------------------------------------------------------------------- #
# ------------------------- 7) Get spectral lines --------------------------- #
# --------------------------------------------------------------------------- #

	df_spectral_lines = get_spectrallines(df=df_filtered,
										  from_sp=from_sp,
										  to_sp=to_sp,
										  save=True)

# --------------------------------------------------------------------------- #
# --------- 8) Merge spectral lines with the continuum to one table --------- #
# --------------------------------------------------------------------------- #

	df_preprocessed = merge_lines_and_continuum(df_spectral_lines, df_continuum)

# ---------------------------------------------------------------------------- #
# ---------------------------- 9) Merge all data ----------------------------- #
# ---------------------------------------------------------------------------- #

# print('DF Preprocessed (Final)')
# print(df_preprocessed.columns)
# print(df_preprocessed.head())

# ---------------------------------------------------------------------------- #
# ------------------------- 10) Run the ML algorithms ------------------------ #
# ---------------------------------------------------------------------------- #

	model = create_model(df_preprocessed, configs['neural_network'])





if __name__ == '__main__':
	main()
