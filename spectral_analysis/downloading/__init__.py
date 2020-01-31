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

def main():
	from_sp = 110001
	to_sp 	= 120000

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
									  save=False)
									
# --------------------------------------------------------------------------- #
# --------- 3) Merge spectra with table containing meta information --------- #
# --------------------------------------------------------------------------- #

	df_merged = merge_with_metatable(from_sp=str(from_sp),
									 to_sp=str(to_sp),
									 save=True,
									 df=None)

if __name__ == '__main__':
	main()
