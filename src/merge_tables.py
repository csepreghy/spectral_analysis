import pickle
import pandas as pd
import time as time


def merge(from_sp, to_sp, save=False):
	"""
	merge()

	Parameters
	----------
	from_sp : string
		The number from which to merge spectra with meta-data. String, beceause it must match
		the filename in folder data/sdss/spectra/
	
	to_sp : string
		The number which specifies the upper limit to merge spectra with meta-data. String, 
		beceause it must match the filename in folder data/sdss/spectra/

	save : boolean
		When True, save the resulting merged table into a pickle
		When False, don't save the resulting merged table
	
	Returns
	-------
	df_merged : pandas.DataFrame
		A merged table that contains spectral data all meta information from 
		data/sdss/meta_table.pkl:
	
		columns:	'flux_list',
					'wavelength',
					'objid',
					'bestObjID',
					'fluxObjID',
					'targetObjID',
					'plate',
					'class',
					'subClass',
					'zErr',
					'petroMag_u',
					'petroMag_g',
					'petroMag_r',
					'petroMag_i',
					'petroMag_z',
					'petroMagErr_u',
					'petroMagErr_g',
					'petroMagErr_r',
					'petroMagErr_i',
					'petroMagErr_z',
					'dec',
					'z',
					'ra'
	"""

	df_spectra = pd.read_pickle('data/sdss/spectra/spectra_' + from_sp + '-' + to_sp + '.pkl')
	df_meta_data = pd.read_pickle('data/sdss/meta_table.pkl')

	df_meta_data["objid"] = df_meta_data['bestObjID'].astype(int)
	df_spectra['objid'] = df_spectra['objid'].astype(int)
	
	df_spectra.drop_duplicates('objid')
	df_meta_data.drop_duplicates()
	
	df_meta_data = df_meta_data.drop(columns={"specObjID"})

	df_merged = pd.merge(df_spectra, df_meta_data, on=['objid'])

	df_merged["dec"] = df_merged['dec_y']
	df_merged["z"] = df_merged['z_y']
	df_merged["ra"] = df_merged['ra_y']
	df_merged = df_merged.drop(columns={'dec_x', 'z_x', 'ra_x', 'dec_y', 'z_y', 'ra_y'})

	if save:
		df_merged.to_pickle('data/sdss/spectra-meta/spectra-meta-merged_' + from_sp + '-' + to_sp + '.pkl')

	return df_merged

def main():
	print('merge_tables.py -- __main__')

	from_sp = '10001'
	to_sp = '20000'

	merge(from_sp, to_sp)

	df = pd.read_pickle('data/sdss/spectra-meta/spectra-meta-merged_' + from_sp + '-' + to_sp + '.pkl')

if __name__ == '__main__':
  	main()