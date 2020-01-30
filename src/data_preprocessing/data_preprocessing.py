import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from skimage import io, filters, feature
from scipy import ndimage
import pickle
from itertools import islice
from tqdm.auto import tqdm

from src.plotify import Plotify

plotify = Plotify()

CUTOFF_MIN = 3850
CUTOFF_MAX = 9100

def apply_gaussian_filter(fluxes, sigma):
  return filters.gaussian(image=fluxes, sigma=sigma)


def plot_one_spectrum(spectra, nth_element, sigma, downsize, filename, save, show_plot):
  gaussian_sigma = sigma
  spectrum_x = spectra.iloc[nth_element]['wavelength']
  spectrum_y = spectra.iloc[nth_element]['flux_list']
  spectrum_title = 'Spectrum with guassian smoothing, sigma = ' + str(gaussian_sigma)
  filename = filename + str(gaussian_sigma) + '.png'

  spectrum_y_filtered = apply_gaussian_filter(spectrum_y, sigma=gaussian_sigma)
  spectrum_y_downsized = spectrum_y_filtered[::downsize]
  spectrum_x = spectrum_x[::downsize]
  print('len(spectrum_y)', len(spectrum_y))

  fig, ax = plotify.plot(x=spectrum_x,
                         y=spectrum_y_downsized,
                         xlabel='Frequencies',
                         ylabel='Flux',
                         title=spectrum_title,
                         figsize=(12, 8),
                         show_plot=show_plot,
                         filename=filename,
                         ymin=np.amin(spectrum_y) - 4,
                         ymax=np.amax(spectrum_y) + 4,
                         save=save
  )

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
	print('Number of rows after filtering: ', str(len(filtered_df)))

	if save:
		filtered_df.to_pickle('filtered_df.pkl')
		# df_filtered.to_msgpack('data/spectra-meta-filtered_0-70k.msg')
		# df_filtered = pd.read_msgpack('data/spectra-meta-filtered_0-70k.msg')
	
	return filtered_df

def spectrum_cutoff(df):
	for index, spectrum in tqdm(df.iterrows(), total=df.shape[0], desc='Spectrum Cutoff: '):
		wavelengths = np.array(spectrum['wavelength'])
		fluxes = np.array(spectrum['flux_list'])

		fluxes = fluxes[(wavelengths > CUTOFF_MIN) & (wavelengths < CUTOFF_MAX)]
		wavelengths = wavelengths[(wavelengths > CUTOFF_MIN) & (wavelengths < CUTOFF_MAX)]

		df.loc[index, 'wavelength'] = [[wavelengths]]
		df.loc[index, 'flux_list'] = [[fluxes]]
		
	print('DF After Cutoff:')
	print(df.columns)
	print(df.head())
	print(f'Length of filtered_df = {len(df)}')

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
    classes = [x.replace('b\'', '').replace('\'', '') for x in classes]
    df['class'] = classes
    df.to_pickle('data/sdss/preprocessed/0-50_preprocessed_2.pkl')

def remove_nested_lists(df):
    """
    remove_nested_lists()

    Takes a dataframe which has two columns with the form [[wavelengths]] and
    [[flux_list]] and removes one of the square brackets. The double brackets
    were a not-so-elegant workaround for a problem in spectrum_cutoff()

    Parameters
    ----------
    df : pandas.DataFrame
        All spectra with the double nested lists [[]]
    
    Returns
    -------

    df : pandas.DataFrame
        The same DataFrame as the input except the double brackets removed
    """
    flux_lists = df['flux_list'].head().to_numpy()
    wavelengths = df['wavelength'].head().to_numpy()

    modified_flux_list = []
    modified_wavelengths = []

    print(flux_lists)

    for flux in tqdm(flux_lists):
        modified_flux_list.append(list(flux[0]))

    for wavelength in tqdm(wavelengths):
        modified_wavelengths.append(list(flux[0]))

    new_df = pd.DataFrame({'flux_list': modified_flux_list,
                           'wavelength': modified_wavelengths})
    
    df = df.drop(columns={'flux_list', 'wavelength'})
    
    for column in df.columns:
        new_df[column] = df[column]

    print(f'{new_df}')

    # df_fluxes = pd.DataFrame({
    #     'flux_list': new_flux_list,
    #     'objid': df['objid'].head()
    # })

    # new_df = df.drop(columns={'flux_list'})
    # new_df.merge(df_fluxes, on='objid')

    # print(f"new_df = {new_df['flux_list']}")
    


def main():
    """
    main()

    Runs a test batch to test whether the functions filter_sources() works properly.
    """

    df_preprocessed = pd.read_pickle('data/sdss/preprocessed/0-50_preprocessed.pkl')
    remove_nested_lists(df_preprocessed)

    # df_spectra = pd.read_pickle('data/sdss/spectra-meta/spectra-meta_1000-2020.pkl')
    # filtered_df = filter_sources(df=df_spectra, save=False)
    # df_cutoff = spectrum_cutoff(filtered_df)
    # print(f'filtered_df = {filtered_df}')


if __name__ == '__main__':
	main()
