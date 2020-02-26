import pandas as pd
import time as time
import numpy as np
import matplotlib.pyplot as plt


from spectral_analysis.data_preprocessing.data_preprocessing import (filter_sources,
                                                                    spectrum_cutoff,
                                                                    create_continuum,
                                                                    merge_lines_and_continuum,
                                                                    remove_nested_lists)

from spectral_analysis.data_preprocessing.merge_tables import merge_with_metatable
from spectral_analysis.data_preprocessing.get_spectrallines import get_spectrallines

# 4) Filter Out Spectra with not enough values ---------------------------- #
# 5) Cut off values from the sides to have the same range for all spectra - #
# 6) Create Continuum that has Gaussian smoothing ------------------------- #
# 7) Get spectral lines --------------------------------------------------- #
# 8) Merge spectral lines with the continuum to one table ----------------- #
# 9) Merge all data ------------------------------------------------------- #

def main():
    from_sp = 50000
    to_sp 	= 100000

# --------------------------------------------------------------------------- #
# --------------- 4) Filter Out Spectra with not enough values -------------- #
# --------------------------------------------------------------------------- #

    df_merged = pd.read_parquet('data/sdss/spectra-meta/50-100_merged.parquet')
    df_filtered = filter_sources(df=df_merged, save=False)
    df_merged = None # To remove from memory

# --------------------------------------------------------------------------- #
# - 5) Cut off values from the sides to have the same range for all spectra - #
# --------------------------------------------------------------------------- #
	
    df_cutoff = spectrum_cutoff(df=df_filtered)
    df_filtered = None # To remove from memory

# --------------------------------------------------------------------------- #
# ------------- 6) Create Continuum that has Gaussian smoothing ------------- #
# --------------------------------------------------------------------------- #

    # df_continuum = create_continuum(df=df_cutoff,
    #                                 sp_index_range=(from_sp, to_sp),
    #                                 sigma=0,
    #                                 downsize=1,
    #                                 save=True)

    # df_continuum = pd.read_pickle('data/sdss/continuum0_50000.pkl')
    remove_nested_lists(df_cutoff, '50_100_preprocessed_original_fluxes')

# --------------------------------------------------------------------------- #
# ------------------------- 7) Get spectral lines --------------------------- #
# --------------------------------------------------------------------------- #

    # df_spectral_lines = get_spectrallines(df=df_cutoff,
    #                                       from_sp=from_sp,
    #                                       to_sp=to_sp,
    #                                       save=True)

# --------------------------------------------------------------------------- #
# --------- 8) Merge spectral lines with the continuum to one table --------- #
# --------------------------------------------------------------------------- #

    # df_preprocessed = merge_lines_and_continuum(df_spectral_lines, df_filtered)
    # df_preprocessed.to_pickle('data/sdss/50-100_spectral_lines_and_fluxes.pkl')

# ---------------------------------------------------------------------------- #
# ---------------------------- 9) Merge all data ----------------------------- #
# ---------------------------------------------------------------------------- #

	# print('DF Preprocessed (Final)')
	# print(df_preprocessed.columns)
	# print(df_preprocessed.head())


if __name__ == "__main__":
	main()


