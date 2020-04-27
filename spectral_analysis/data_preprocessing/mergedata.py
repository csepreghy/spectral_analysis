from __future__ import division
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from tqdm.auto import tqdm

from spectral_analysis.data_preprocessing.data_preprocessing import get_fluxes_from_h5, get_wavelengths_from_h5

def merge_data(filenames):
    data = pd.read_parquet('data/sdss/spectra-meta/final/' + filenames[0])
    
    for i in tqdm(range(len(filenames))):
        x = pd.read_parquet('data/sdss/spectra-meta/final/' + filenames[i])
        print(f'x = {type(x)}')
        datalist = pd.concat([data, x], ignore_index=True)
        data = datalist

    return data

def merge_hdf5_files(filenames, output):
    """
    merge_hdf5_files()
    
    Takes a list of hdf5 files, merges them and saves the result

    Parameters
    ----------

    filenames : [str, str., ..., str]
        List of filenames that will be merged

    output : str
        Name of file to be saved
    """


    wavelengths = get_wavelengths_from_h5(filename=filenames[0])
    fluxes1 = get_fluxes_from_h5(filename=filenames[0])
    print(f'fluxes1.shape = {fluxes1.shape}')
    fluxes2 = get_fluxes_from_h5(filename=filenames[1])
    print(f'fluxes2.shape = {fluxes2.shape}')

    merged_fluxes = np.concatenate((fluxes1, fluxes2), axis=0)
    print(f'merged_fluxes = {merged_fluxes}')
    print(f'merged_fluxes.shape = {merged_fluxes.shape}')



    # store = pd.HDFStore(data_path + filename)
    # store.put('spectral_data', df, format='fixed', data_columns=True)
    # store.put('fluxes', flux_df, format='fixed', data_columns=True)
    # store.put('wavelengths', wavelength_df)

    # print(store.keys())

    # store.close()

def main():
	path = 'data/sdss/spectra-meta/final'
	filenames = [f for f in listdir(path) if isfile(join(path, f))]
	try: filenames.remove('.DS_Store')
	except: print('.DS_Store is not in folder')
	
	print(f'filenames = {filenames}')

	all_data = merge_data(filenames)
	print(f'all_data = {all_data}')
	all_data.to_parquet('data/sdss/balanced.parquet')

    # merge_hdf5_files(['/sdss/preprocessed/0-50_original_fluxes.h5',
    #                   '/sdss/preprocessed/50-100_original_fluxes.h5'],
    #                   '/sdss/preprocessed/0-100_original_fluxes.h5')

	# df = pd.read_parquet('data/sdss/spectra-meta/0-50_merged.parquet')
	# print('df parquet', df)

if __name__ == "__main__":
	main()
