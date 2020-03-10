import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pickle
from itertools import islice
from tqdm.auto import tqdm

from skimage import io, filters, feature
from scipy import ndimage, interpolate

from spectral_analysis.plotify import Plotify

import pathlib
import os

plotify = Plotify()

CUTOFF_MIN = 3850
CUTOFF_MAX = 9100

def apply_gaussian_filter(fluxes, sigma):
    """
    apply_gaussian_filter()

    Takes one spectrum convolves a gaussian filter with it and returns the resulting list of fluxes

    Parameters
    ----------
    fluxes : numpy.ndarray
        1D numpy array of a list of fluxes
    
    sigma : int
        the sigma with which the gaussian convolving is applied
    
    Returns
    -------
    fluxes : numpy.ndarray
        1D numpy array that results from convolving the gaussian filter
    """
    return filters.gaussian(image=fluxes, sigma=sigma)


def plot_spectrum(fluxes, wavelengths, save=False, filename=None):
    """
    plot_spectrum()

    Takes a set of fluxes and their corresponding wavelengths and plots them with an option
    to save the resulting plot.

    Parameters
    ----------
    fluxes : numpy.ndarray
        A list of fluxes that will be ploted
    
    wavelengths : numpy.ndarray
        A list of wavelengths that correspond to the fluxes
    
    save : boolean (optional)
        When True, saves plot
        When False, does not save plot
    
    filename : str (optional)
        The name of the file that will be saved.
    """

    spectrum_title = 'Spectrum'
    
    if filename is not None: filename = filename + '.png'
    else: filename = ''

    print('len(wavelengths)', len(wavelengths))
    print('len(fluxes)', len(fluxes))

    fig, ax = plotify.plot(x=wavelengths,
                           y=fluxes,
                           xlabel='Frequencies',
                           ylabel='Flux',
                           title=spectrum_title,
                           figsize=(12, 8),
                           show_plot=True,
                           filename=filename,
                           save=save)

def create_continuum(df, sp_index_range, sigma, downsize, save):
	"""
	create_continuum()

	Takes all spectra as a DataFrame and from the flux_list it applies gaussian smoothing to get dampen
	fluctuations of the spectrum and to reduce noise in data. Then it reduces the number of datapoints
	in the spectrum by a fraction given as a parameter. This helps in dimensionality reduction for the 
	ML algorithms.

	Parameters
	----------
	df : pandas.DataFrame
		All unfiltered spectra that is already merged with the metatable
	
	sp_index_range : (number, number)
		The upper and lower bound of the indexes that show from which index to which index the function
		should consider spectra. Only used for filenames in saving

	sigma : number
		The sigma to use for applying Gaussian smoothing to the spectrum. Typical values range from 2 to 32.

	downsize : number
		The fraction to which to reduce the number of datapoints in the spectrum continuum. This helps
		in dimensionality reduction for the ML algorithms.

	save : boolean
		When True, saves the filtered DataFrame into a pickle file
		When False, doesn't save

	Returns
	-------
	df_continuum : pandas.DataFrame
	"""

	rows_after_smoothing = []

	for index, spectrum in tqdm(df.iterrows(), total=df.shape[0], desc='Create Continuum: '):
		wavelengths = spectrum['wavelength']
		fluxes = np.array(spectrum['flux_list'])
		fluxes_filtered = apply_gaussian_filter(fluxes=fluxes, sigma=sigma)
		fluxes_downsized = fluxes_filtered[::downsize]
		wavelengths_downsized = wavelengths[::downsize]
	
		row = {
			'wavelength': wavelengths_downsized,
			'flux_list': fluxes_downsized,
			'petroMagErr_u': spectrum['petroMagErr_u'],
			'petroMagErr_g': spectrum['petroMagErr_g'],
			'petroMagErr_r': spectrum['petroMagErr_r'],
			'petroMagErr_i': spectrum['petroMagErr_i'],
			'petroMagErr_z': spectrum['petroMagErr_z'],
			'petroMag_u': spectrum['petroMag_u'],
			'petroMag_g': spectrum['petroMag_g'],
			'petroMag_r': spectrum['petroMag_r'],
			'petroMag_i': spectrum['petroMag_i'],
			'petroMag_z': spectrum['petroMag_z'],
			'subClass': str(spectrum['subClass']),
			'objid': spectrum['objid'],
			'plate': spectrum['plate'],
			'class': str(spectrum['class']),
			'zErr': spectrum['zErr'],
			'dec': spectrum['dec'],
			'ra': spectrum['ra'],
			'z': spectrum['z'],
		}

		rows_after_smoothing.append(row)
    
	df_continuum = pd.DataFrame(rows_after_smoothing)

	if save:
		filename = 'data/sdss/continuum' + str(sp_index_range[0]) + '_' + str(sp_index_range[1]) + '.pkl'
		df_continuum.to_pickle(filename)
	
	print('DF Continuun: ')
	print(df_continuum.columns)
	print(df_continuum)
	print(f'Length of df_continuum = {len(df_continuum)}')
  
	return df_continuum

# spectra = pd.read_pickle('data/sdss/spectra-meta/spectra-meta-merged_5001-10000.pkl')
# continuum_df = create_continuum(df=spectra, sigma=8, downsize=2)

# print('continuum_df', continuum_df)

# plot_one_spectrum(spectra=continuum_df, nth_element=2300, sigma=4, downsize=4, filename=('animation_plot'), save=False, show_plot=True)

def filter_sources(df, save=False):
    """
    filter_sources()

    Takes all spectra as a DataFrame and removes all sources that fall out of the optimal wavelength range
    and therefore don't have enough values for classification

    Parameters
    ---------
    df : pandas.DataFrame
        All unfiltered spectra that is already merged with the metatable

    save : boolean
        When True, saves the filtered DataFrame into a pickle file
        When False, doesn't save

    Returns
    -------
    df_filtered : pandas.DataFrame
    """

    print(f'df.shape[0] = {df.shape[0]}')
    duplicates = df[df.duplicated(subset=['objid', 'z'])]
    df = df.drop_duplicates(subset=['objid', 'z'])
    print(f'duplicates = {duplicates}')
    print(f'df.shape[0] = {df.shape[0]}')

    rows_after_removal = []

    print('Number of rows before filtering: ', str(df.shape[0]))
    print('df', df.columns)

    for index, spectrum in tqdm(df.iterrows(), total=df.shape[0], desc='Filtering Sources: '):
        min_value = np.amin(spectrum['wavelength'].tolist())
        max_value = np.amax(spectrum['wavelength'].tolist())

        if min_value < CUTOFF_MIN and max_value > CUTOFF_MAX:
            row = {'wavelength': spectrum['wavelength'].tolist(),
                    'flux_list': spectrum['flux_list'].tolist(),
                    'petroMagErr_u': spectrum['petroMagErr_u'],
                    'petroMagErr_g': spectrum['petroMagErr_g'],
                    'petroMagErr_r': spectrum['petroMagErr_r'],
                    'petroMagErr_i': spectrum['petroMagErr_i'],
                    'petroMagErr_z': spectrum['petroMagErr_z'],
                    'petroMag_u': spectrum['petroMag_u'],
                    'petroMag_g': spectrum['petroMag_g'],
                    'petroMag_r': spectrum['petroMag_r'],
                    'petroMag_i': spectrum['petroMag_i'],
                    'petroMag_z': spectrum['petroMag_z'],
                    'subClass': spectrum['subClass'],
                    'fluxObjID': spectrum['fluxObjID'],
                    'objid': spectrum['objid'],
                    'plate': spectrum['plate'],
                    'class': spectrum['class'],
                    'zErr': spectrum['zErr'],
                    'dec': spectrum['dec'],
                    'ra': spectrum['ra'],
                    'z': spectrum['z']}

        rows_after_removal.append(row)
        
    filtered_df = pd.DataFrame(rows_after_removal)
    print(f'filtered_df = {filtered_df}')
    print('Number of rows after filtering: ', str(len(filtered_df)))

    if save:
        filtered_df.to_pickle('filtered_df.pkl')
        # df_filtered.to_msgpack('data/spectra-meta-filtered_0-70k.msg')
        # df_filtered = pd.read_msgpack('data/spectra-meta-filtered_0-70k.msg')

    return filtered_df

def spectrum_cutoff(df, save=False):
    for index, spectrum in tqdm(df.iterrows(), total=df.shape[0], desc='Spectrum Cutoff: '):
        wavelengths = np.array(spectrum['wavelength'])
        fluxes = np.array(spectrum['flux_list'])

        fluxes = fluxes[(wavelengths > CUTOFF_MIN) & (wavelengths < CUTOFF_MAX)]
        wavelengths = wavelengths[(wavelengths > CUTOFF_MIN) & (wavelengths < CUTOFF_MAX)]

        df.loc[index, 'wavelength'] = [[wavelengths]]
        df.loc[index, 'flux_list'] = [[fluxes]]
        

    print('DF After Cutoff:')
    print(df.columns)
    print(df)
    print(f'Length of cutoff_df = {len(df)}')

    if save == True:
        df.to_pickle('data/sdss/50-100_spectrum_cutoff.pkl')

    return df

def check_minmax_values(spectra, sigma=16, downsize=8):
  min_wavelength_values = []
  max_wavelength_values = []

  for index, spectrum in islice(spectra.iterrows(), 500):
    min_wavelength_values.append(np.amin(spectrum['wavelength'].tolist()))
    max_wavelength_values.append(np.amax(spectrum['wavelength'].tolist()))

  absolute_min = np.amax(np.array(min_wavelength_values))
  absolute_max = np.amin(np.array(max_wavelength_values))

  print('absolute_min', absolute_min)
  print('absolute_max', absolute_max)
  print('min_wavelength_values', min_wavelength_values)

  plotify.plot(
    y=min_wavelength_values,
    x=range(len(min_wavelength_values)),
    title='Minimum Wavelength Values',
    ymin=3400,
    ymax=6500,
    xlabel='',
    ylabel='Wavelength',
    filename='minimum-wavelength-values'
  )

  plotify.plot(
    y=max_wavelength_values,
    x=range(len(max_wavelength_values)),
    title='Maximum Wavelength Values',
    ymin=6000,
    ymax=14000,
    xlabel='',
    ylabel='Wavelength',
    filename='maximum-wavelength-values'
  )

def clear_duplicates(df1, df2):
	# Get the IDs from both data frames
	id_1 = df1['objid'].get_values()
	id_2 = df2['objid'].get_values()

	# Make a list with the duplicate IDs
	u, c = np.unique(id_1, return_counts=True)
	dup = u[c > 1]

	# Get the indices
	indices = []
	for n in dup:
		indices.append(list(id_1).index(n))

	# Drop double IDs
	df1_new = df1.drop(indices)
	df2_new = df2.drop(indices)

	# Reset index
	df1_new = df1_new.reset_index()
	df2_new = df2_new.reset_index()
	
	return df1_new, df2_new

def merge_lines_and_continuum(spectral_lines, continuum):
    """
    # Function to check if the IDs are unique:
    def allUnique(x):
        seen = set()
        return not any(i in seen or seen.add(i) for i in x)
    """

    print(f'continuum = {continuum}')

    # First round clearing for duplicates
    spectral_lines2, continuum2 = clear_duplicates(spectral_lines, continuum)

    # Second round clearing for triple duplicates
    spectral_lines3, continuum3 = clear_duplicates(spectral_lines2, continuum2)

    # Merge the spectral lines and continuum table on objID
    df_merge = continuum3.merge(spectral_lines3, on='objid')

	# Convert the specclass bytes into strings
	# specclass_bytes = df_merge['class'].get_values()
	# specclass = []
	# for i in specclass_bytes:
	# 	specclass.append(i.decode("utf-8"))
		
	# specclass = np.array(specclass)

	# df_merge['class'] = specclass

	# Order the columns in a more sensible way
    df_merge = df_merge[['objid',
                         'flux_list',
                         'wavelength',
                         'spectral_lines',
                         'z',
                         'zErr',
                         'ra',
                         'dec',
                         'plate',
                         'class',
                         'subClass',
                         'petroMagErr_u',
                         'petroMagErr_g',
                         'petroMagErr_r',
                         'petroMagErr_i',
                         'petroMagErr_z',
                         'petroMag_u',
                         'petroMag_g',
                         'petroMag_r',
                         'petroMag_i',
                         'petroMag_z']]

    return df_merge

def remove_bytes_from_class(df):
    classes = df['class'].to_numpy()
    print(f'classes = {classes}')
    classes = [str(x).replace('b\'', '').replace('\'', '') for x in classes]
    
    df['class'] = classes

    return df

def expand_list(df, list_column, new_column): 
    lens_of_lists = df[list_column].apply(len)
    origin_rows = range(df.shape[0])
    destination_rows = np.repeat(origin_rows, lens_of_lists)
    non_list_cols = (
      [idx for idx, col in enumerate(df.columns)
       if col != list_column]
    )
    expanded_df = df.iloc[destination_rows, non_list_cols].copy()
    expanded_df[new_column] = (
      [item for items in df[list_column] for item in items]
      )
    expanded_df.reset_index(inplace=True, drop=True)
    return expanded_df

def remove_nested_lists(df, filename):
    """
    remove_nested_lists()

    Takes a dataframe which has two columns with the form [[wavelengths]] and
    [[flux_list]] and removes one of the square brackets. The double brackets
    were a not-so-elegant workaround for a problem in spectrum_cutoff(). Then
    it saves everything into an hdf5 file into 3 tables.
        - spectral_data: all 

    Parameters
    ----------
    df : pandas.DataFrame
        All spectra with the double nested lists [[]]
    
    filename : str
        filename that will be saved in HDF5
    
    Returns
    -------

    df : pandas.DataFrame
        The same DataFrame as the input except the double brackets removed
    """


    flux_lists = df['flux_list'].to_numpy()
    wavelengths = df['wavelength'].to_numpy()

    modified_flux_list = []
    modified_wavelengths = []

    for flux in tqdm(flux_lists):
        modified_flux_list.append(flux[0])

    for wavelength in tqdm(wavelengths):
        modified_wavelengths.append(wavelength[0])

    # new_df = pd.DataFrame({'flux_list': modified_flux_list})

    df = df.drop(columns={'flux_list', 'wavelength'})

    flux_column_list = []
    for flux_column in range(len(flux_lists[0][0])):
        flux_column_list.append(f'flux_str{(flux_column)}')
    
    wavelength_df = pd.DataFrame({'wavelengths': modified_wavelengths[0]})
    print(f'wavelength_df = {wavelength_df}') 
    
    flux_df = pd.DataFrame({'objid': df['objid']})
    flux_df[flux_column_list] = pd.DataFrame(modified_flux_list, index=None, columns=flux_column_list)

    print(f'modified_flux_list = {len(modified_flux_list)}')
    print(f'flux_column_list = {flux_column_list}')
    print(f'flux_df.values = {flux_df}')
    print(f'df spectral_data = {df}')

    data_path = '/Users/csepreghyandras/the_universe/projects/spectral-analysis/data/sdss/preprocessed/'

    store = pd.HDFStore(data_path + filename)
    store.put('spectral_data', df, format='fixed', data_columns=True)
    store.put('fluxes', flux_df, format='fixed', data_columns=True)
    store.put('wavelengths', wavelength_df)

    print(store.keys())

    store.close()
    
def get_fluxes_from_h5(filename):
    """

    """
    filepath = '/Users/csepreghyandras/the_universe/projects/spectral-analysis/data/' + filename
    flux_df = pd.read_hdf(filepath, key='fluxes')

    fluxes = flux_df.values
    print(f'fluxes = {type(fluxes[0])}')

    return fluxes

def get_wavelengths_from_h5(filename):
    """

    """
    filepath = '/Users/csepreghyandras/the_universe/projects/spectral-analysis/data/' + filename
    wavelengths_df = pd.read_hdf(filepath, key='wavelengths')

    wavelengths = wavelengths_df.values
    print(f'wavelengths = {type(wavelengths[0])}')

    return wavelengths

def apply_gaussian_to_fluxes(fluxes, sigma):
    gaussian_fluxes = np.zeros_like(fluxes)

    for index, flux in tqdm(enumerate(fluxes), total=len(gaussian_fluxes), desc='Applying Gaussian Filter: '):
        gaussian_flux = apply_gaussian_filter(flux, sigma)
        gaussian_fluxes[index] = gaussian_flux
    
    return gaussian_fluxes

def merge_spectral_lines_with_hdf5_data(df_source_info, df_spectral_lines):
    spectral_lines_columns = []

    spectral_lines = df_spectral_lines.values

    for i in range(len(spectral_lines[0][1])):
        spectral_lines_columns.append(f'spectral_line_{i}')
    
    spectral_lines_objids = spectral_lines[:, 0]
    spectral_lines_lists = spectral_lines[:, 1]
    spectral_lines_values = []
    
    for i in spectral_lines_lists:
        row = []
        for x in i:
            row.append(x)

        spectral_lines_values.append(row)


    df_spectral_lines_expanded = pd.DataFrame({'objid': spectral_lines_objids})
    df_spectral_lines_expanded[spectral_lines_columns] = pd.DataFrame(spectral_lines_values, columns=spectral_lines_columns)
    df_merged = pd.concat([df_source_info, df_spectral_lines_expanded], axis=1)
    df_merged = df_merged.loc[:, ~df_merged.columns.duplicated()]

    print(f'df_merged = {df_merged}')
    print(f'df_merged.columns = {df_merged.columns}')

    data_path = '/Users/csepreghyandras/the_universe/projects/spectral-analysis/data/sdss/preprocessed/'
    filename = '50-100_original_fluxes_speclines.h5'

    flux_df = pd.read_hdf('data/sdss/preprocessed/50-100_original_fluxes.h5', key='fluxes')
    wavelength_df = pd.read_hdf('data/sdss/preprocessed/50-100_original_fluxes.h5', key='wavelengths')

    print(f'flux_df = {flux_df}')
    print(f'spectral_lines_expanded = {df_spectral_lines_expanded}')

    store = pd.HDFStore(data_path + filename)
    store.put('spectral_data', df_merged, format='fixed', data_columns=True)
    store.put('fluxes', flux_df, format='fixed', data_columns=True)
    store.put('wavelengths', wavelength_df)

    print(store.keys())

    store.close()

def interpolate_and_reduce_to(df_fluxes, df_source_info, df_wavelengths, filename, reduce_to=1536):
    fluxes = np.delete(df_fluxes.values, 0, axis=1)
    objids = df_fluxes.values[:,0]
    wavelengths = df_wavelengths.values.flatten()

    new_wavelengths = np.linspace(min(wavelengths), max(wavelengths), reduce_to)
    f = interpolate.interp1d(wavelengths, fluxes, kind='zero')
    new_fluxes = f(new_wavelengths)

    flux_column_list = []
    for flux_column in range(len(new_fluxes[0])):
        flux_column_list.append(f'flux_{str(flux_column)}')

    # plot_spectrum(new_fluxes[24], new_wavelengths)

    df_new_wavelengths = pd.DataFrame({'wavelengths': new_wavelengths})
    df_new_fluxes = pd.DataFrame({'objid': objids})
    df_new_fluxes[flux_column_list] = pd.DataFrame(new_fluxes, index=None, columns=flux_column_list)

    print(f'df_new_fluxes = {df_new_fluxes}')
    print(f'wavelengths_df = {df_new_wavelengths}')

    data_path = '/Users/csepreghyandras/the_universe/projects/spectral-analysis/data/sdss/preprocessed/'

    store = pd.HDFStore(data_path + filename)
    store.put('source_info', df_source_info, format='fixed', data_columns=True)
    store.put('fluxes', df_new_fluxes, format='fixed', data_columns=True)
    store.put('wavelengths', df_new_wavelengths)


def main():
    df_fluxes = pd.read_hdf('data/sdss/preprocessed/0-50_i_fluxes_1536.h5', key='fluxes')
    df_source_info = pd.read_hdf('data/sdss/preprocessed/0-50_i_fluxes_1536.h5', key='spectral_data')
    df_wavelengths = pd.read_hdf('data/sdss/preprocessed/0-50_i_fluxes_1536.h5', key='wavelengths')

    # interpolate_and_reduce_to(df_fluxes, df_source_info, df_wavelengths, '0-50_i_fluxes_1536.h5')

if __name__ == '__main__':
	main()