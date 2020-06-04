import pandas as pd
import time as time
import numpy as np
import matplotlib.pyplot as plt


from spectral_analysis.spectral_analysis.data_preprocessing.data_preprocessing import (filter_sources,
                                                                    spectrum_cutoff,
                                                                    create_continuum,
                                                                    merge_lines_and_continuum,
                                                                    remove_nested_lists,
                                                                    merge_spectral_lines_with_hdf5_data,
                                                                    convert_byte_classes)

from spectral_analysis.spectral_analysis.data_preprocessing.merge_tables import merge_with_metatable
from spectral_analysis.spectral_analysis.data_preprocessing.get_spectrallines import get_spectrallines

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

    # print(f'df_merged 1 = {df_merged}')
    # df_merged = df_merged.drop_duplicates(subset='objid', keep='first', inplace=False)
    # df_filtered = filter_sources(df=df_merged, save=False)

    # df_merged = None # To remove from memory
    
    # print(f'df_filtered = {df_filtered}')
    # print(f'df_merged = {df_merged}')

# --------------------------------------------------------------------------- #
# - 5) Cut off values from the sides to have the same range for all spectra - #
# --------------------------------------------------------------------------- #
	
    # df_filtered = pd.read_parquet('data/sdss/balanced_uniques.parquet')
    # df_cutoff = spectrum_cutoff(df=df_filtered, save=False)
    # df_cutoff.to_parquet('data/sdss/balanced_cutoff.parquet')
    # df_filtered = None # To remove from memory
    # print(f'y = {y}')

# --------------------------------------------------------------------------- #
# ------------- 6) Create Continuum that has Gaussian smoothing ------------- #
# --------------------------------------------------------------------------- #

    # df_continuum = create_continuum(df=df_cutoff,
    #                                 sp_index_range=(from_sp, to_sp),
    #                                 sigma=0,
    #                                 downsize=1,
    #                                 save=True)

    # df_cutoff = pd.read_parquet('data/sdss/balanced_cutoff.parquet')
    # print(f'df_cutoff = {df_cutoff}')
    # remove_nested_lists(df_cutoff, 'balanced.h5')

# --------------------------------------------------------------------------- #
# ------------------------- 7) Get spectral lines --------------------------- #
# --------------------------------------------------------------------------- #



    # df_source_info['class'] = [x.decode('utf-8') for x in df_source_info['class']]


    df_fluxes =  pd.read_hdf('data/sdss/preprocessed/balanced_spectral_lines.h5', key='fluxes')
    df_source_info = pd.read_hdf('data/sdss/preprocessed/balanced_spectral_lines.h5', key='source_info')
    df_wavelengths = pd.read_hdf('data/sdss/preprocessed/balanced_spectral_lines.h5', key='wavelengths')



    # df_spectral_lines = get_spectrallines(df_fluxes=df_fluxes,
    #                                       df_source_info=df_source_info,
    #                                       df_wavelengths=df_wavelengths,
    #                                       from_sp=0,
    #                                       to_sp=96115,
    #                                       save=True)

    # df_spectral_lines = pd.read_pickle('data/sdss/spectral_lines/spectral_lines_0_96115.ecpkl')

    # convert_byte_classes(df_fluxes, df_source_info, df_wavelengths)

    quasars = df_source_info.loc[df_source_info['class'] == 'QSO']
    print(f'len(quasars) = {len(quasars)}')

    galaxies = df_source_info.loc[df_source_info['class'] == 'GALAXY']
    print(f'len(galaxies) = {len(galaxies)}')

    stars = df_source_info.loc[df_source_info['class'] == 'STAR']
    print(f'len(stars) = {len(stars)}')

    print(f'df_fluxes = {df_fluxes}')
    print(f'df_source_info = {df_source_info}')

    print(f'df_fluxes.columns = {df_fluxes.columns}')
    print(f'df_source_info.columns = {df_source_info.columns}')
    print(f'len = df_source_info.columns = {len(df_source_info.columns)}')

    print(f'df_wavelengths = {df_wavelengths}')


# ---------------------------------------------------------------------------- #
# ---------------------------- 9) Merge all data ----------------------------- #
# ---------------------------------------------------------------------------- #

	# print('DF Preprocessed (Final)')
	# print(df_preprocessed.columns)
	# print(df_preprocessed.head())


if __name__ == "__main__":
	main()

