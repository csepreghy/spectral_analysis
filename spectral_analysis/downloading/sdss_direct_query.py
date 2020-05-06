from astroquery.sdss import SDSS
import pandas as pd
import time as time

def truncate(n):
  return int(n * 1000) / 1000

def get_coordinates_from_query(save_metatable=False, save_coordinates=False, source_type=None):
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

    if source_type == 'QSO':
        query = "select \
                    spec.z, spec.ra, spec.dec, spec.specObjID, spec.bestObjID, spec.fluxObjID, spec.targetObjID, spec.plate, \
                    spec.class, spec.subClass, spec.zErr, spho.petroMag_u, spho.petroMag_g, spho.petroMag_r, spho.petroMag_i, \
                    spho.petroMag_z, spho.petroMagErr_u, spho.petroMagErr_g, spho.petroMagErr_r, spho.petroMagErr_i, spho.petroMagErr_z \
                    from SpecObjAll AS spec \
                    JOIN SpecPhotoAll AS spho ON spec.specObjID = spho.specObjID \
                    where \
                    spec.zWarning = 0 AND spec.class = 'QSO'"

    elif source_type == 'STAR':
        query = "select \
                    spec.z, spec.ra, spec.dec, spec.specObjID, spec.bestObjID, spec.fluxObjID, spec.targetObjID, spec.plate, \
                    spec.class, spec.subClass, spec.zErr, spho.petroMag_u, spho.petroMag_g, spho.petroMag_r, spho.petroMag_i, \
                    spho.petroMag_z, spho.petroMagErr_u, spho.petroMagErr_g, spho.petroMagErr_r, spho.petroMagErr_i, spho.petroMagErr_z \
                    from SpecObjAll AS spec \
                    JOIN SpecPhotoAll AS spho ON spec.specObjID = spho.specObjID \
                    where \
                    spec.zWarning = 0 AND spec.class = 'STAR'"

    else:
        query = "SELECT\
                    spec.z, spec.ra, spec.dec, spec.specObjID, spec.bestObjID, spec.fluxObjID, spec.targetObjID, spec.plate, \
                    spec.class, spec.subClass, spec.zErr, spho.petroMag_u, spho.petroMag_g, spho.petroMag_r, spho.petroMag_i, \
                    spho.petroMag_z, spho.petroMagErr_u, spho.petroMagErr_g, spho.petroMagErr_r, spho.petroMagErr_i, spho.petroMagErr_z, \
                    em.Flux_Hb_4861, em.Flux_Hb_4861_Err, em.Amplitude_Hb_4861, em.Amplitude_Hb_4861_Err, \
                    em.Flux_OIII_4958, em.Flux_OIII_4958_Err, em.Amplitude_OIII_4958, em.Amplitude_OIII_4958_Err, \
                    em.Flux_OIII_5006, em.Flux_OIII_5006_Err, em.Amplitude_OIII_5006, em.Amplitude_OIII_5006_Err, \
                    em.Flux_Ha_6562, em.Flux_Ha_6562_Err, em.Amplitude_Ha_6562, em.Amplitude_Ha_6562_Err, \
                    em.Flux_NII_6547, em.Flux_NII_6547_Err, em.Amplitude_NII_6547, em.Amplitude_NII_6547_Err, \
                    em.Flux_NII_6583, em.Flux_NII_6583_Err, em.Amplitude_NII_6583, em.Amplitude_NII_6583_Err \
                    FROM SpecObjAll AS spec \
                    JOIN SpecPhotoAll AS spho ON spec.specObjID = spho.specObjID \
                    JOIN emissionLinesPort AS em ON em.specObjID = spec.specObjID \
                    WHERE \
                    spec.zWarning = 0 AND spec.class = 'AGN'"

    res = SDSS.query_sql(query, timeout=500)
    df = res.to_pandas()
    print('df', df)
    print(df.columns)

    # df_coordinate_list = pd.DataFrame(df["ra"])
    # df_coordinate_list["dec"]=df["dec"]

    # print(f'df_coordinate_list = {df_coordinate_list}')
    # print(f'df = {df}')

    # if save_coordinates:
    #     df_coordinate_list.to_csv('data/sdss/star_coordinate_list.csv')

    if save_metatable:
        df.to_pickle('data/sdss/qso_meta_table_emissionlines.pkl')

    end = time.clock()
    tt = end - start
    print("time consuming:", truncate(tt), 's')

def main():
	"""
	main()

	Runs a test coordinate download.
	"""
	print('sdss_direct_query.py -- __main__')

	get_coordinates_from_query(save_metatable=True, save_coordinates=False)

if __name__ == '__main__':
	main()