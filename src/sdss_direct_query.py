from astroquery.sdss import SDSS
import pandas as pd
import time as time

def truncate(n):
  return int(n * 1000) / 1000

def get_coordinates_from_query(save_metatable=False, save_coordinates=False):
	"""
	get_coordinates_from_query()

	Downloads and saves into CSV a list of coordinates based on the SQL query written
	in the function

	Parameters
	----------
	save_metatable : boolean
		When True, save the resulting DataFrame containing meta data into a pickle
		When False, don't save
	save_coordinates : boolean
		When True, save the coordinates to a CSV
		When False, don't save

	"""
	start = time.clock()

	query = "select \
	spec.z, spec.ra, spec.dec, spec.specObjID, spec.bestObjID, spec.fluxObjID, spec.targetObjID, spec.plate, spec.class, spec.subClass, spec.zErr, \
	spho.petroMag_u, spho.petroMag_g,	spho.petroMag_r,	spho.petroMag_i,	spho.petroMag_z,	spho.petroMagErr_u, spho.petroMagErr_g, spho.petroMagErr_r, spho.petroMagErr_i, spho.petroMagErr_z \
	from SpecObjAll AS spec \
	JOIN SpecPhotoAll AS spho ON spec.specObjID = spho.specObjID \
	where \
	spec.zWarning = 0"

	res = SDSS.query_sql(query)
	df = res.to_pandas()
	print('df.head()', df.head())

	df_coordinate_list = pd.DataFrame(df["ra"])
	df_coordinate_list["dec"]=df["dec"]

	print(f'df_coordinate_list = {df_coordinate_list}')
	print(f'df = {df}')

	if save_coordinates:
		df_coordinate_list.to_csv('data/sdss/coordinate_list.csv')
	
	if save_metatable:
		df.to_pickle('data/sdss/meta_table.pkl')
	
	end = time.clock()
	tt = end - start
	print("time consuming:", truncate(tt), 's')

def main():
	"""
	main()

	Runs a test coordinate download.
	"""
	print('sdss_direct_query.py -- __main__')

	get_coordinates_from_query(save_metatable=False, save_coordinates=False)

if __name__ == '__main__':
	main()