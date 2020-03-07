import pandas as pd
import time as time
import numpy as np
import matplotlib.pyplot as plt

from sdss_direct_query import get_coordinates_from_query
from downloading import download_spectra
from spectral_analysis.data_preprocessing.data_preprocessing import filter_sources, spectrum_cutoff, create_continuum, merge_lines_and_continuum
from spectral_analysis.data_preprocessing.merge_tables import merge_with_metatable

from spectral_analysis.data_preprocessing.get_spectrallines import get_spectrallines

# 1) Get coordinates from query ------------------------------------------- #
# 2) Download data -------------------------------------------------------- # 
# 3) Merge spectra with table containing meta information ----------------- #

def main():
    from_sp = 0
    to_sp 	= 10000

    # --------------------------------------------------------------------------- #
    # ----------------------- 1) Get cordinates from query ---------------------- #
    # --------------------------------------------------------------------------- #

    # get_coordinates_from_query(save_metatable=True, save_coordinates=True)

    # --------------------------------------------------------------------------- #
    # ----------------------------- 2) Download Data ---------------------------- #
    # --------------------------------------------------------------------------- #
    # coord_list_url = str('data/star_coordinate_list.csv')
    # df_raw_spectra = download_spectra(coord_list_url=coord_list_url,
    #                                   from_sp=from_sp,
    #                                   to_sp=to_sp,
    #                                   save=False)

    # --------------------------------------------------------------------------- #
    # --------- 3) Merge spectra with table containing meta information --------- #
    # --------------------------------------------------------------------------- #

    # df_merged = merge_with_metatable(from_sp=str(from_sp),
    #                                  to_sp=str(to_sp),
    #                                  save=True,
    #                                  df_spectra=df_raw_spectra)

    df = pd.read_pickle('data/sdss/spectra-meta/spectra-meta_0-10000.pkl')
    print(f'df = {df}')

if __name__ == '__main__':
	main()
