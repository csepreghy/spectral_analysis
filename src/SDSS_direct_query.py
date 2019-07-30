from astroquery.sdss import SDSS
import pandas as pd
import time as time

def truncate(n):
  return int(n * 1000) / 1000

def query():
  start = time.clock()

  query = "select \
  spec.z, spec.ra, spec.dec, spec.specObjID, spec.plate, spec.class, spec.subClass, spec.zErr, \
  spho.petroMag_u, spho.petroMag_g,	spho.petroMag_r,	spho.petroMag_i,	spho.petroMag_z,	spho.petroMagErr_u, spho.petroMagErr_g, spho.petroMagErr_r, spho.petroMagErr_i, spho.petroMagErr_z \
  from SpecObjAll AS spec\
  JOIN SpecPhotoAll AS spho ON spec.specObjID = spho.specObjID \
  where \
  spec.zWarning = 0"

  res = SDSS.query_sql(query)
  df = res.to_pandas()
  print('df.head()', df.head())

  df_coordinate_list = pd.DataFrame(df["ra"])
  df_coordinate_list["dec"]=df["dec"]

  df_coordinate_list.to_csv('data/sdss/coordinate_list.csv')
  end = time.clock()
  df.to_pickle('data/sdss/specobj_specphoto_all.pkl')
  tt = end - start
  print("time consuming:", truncate(tt), 's')


query()
