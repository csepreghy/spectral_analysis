import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import time as time
import pickle

from astropy.table import Table
from astropy import coordinates as coords
import astropy.units as u
from astroquery.sdss import SDSS

def download_spectra(coord_list_url, from_sp, to_sp):
  t_start = time.clock()

  coord_list = pd.read_csv(coord_list_url)
  print('coord_list.head()', coord_list.head())

  ra_list = coord_list["ra"].tolist()
  dec_list = coord_list["dec"].tolist()

  ra = ra_list[from_sp:to_sp]
  dec = dec_list[from_sp:to_sp]

  n_errors = 0

  df = {}
  df['flux_list'] = []
  df['wavelength'] = []
  df['z'] = []
  df['ra'] = []
  df['dec'] = []
  df['objid'] = []

  n_coordinates = len(ra)
  number_none = 0

  for i in range(n_coordinates):
    try:
      pos = coords.SkyCoord((ra[i]) * u.deg, (dec[i]) * u.deg, frame='icrs')
      xid = SDSS.query_region(pos, spectro=True) # radius=5 * u.arcsec)

      if xid == None:
        number_none = number_none + 1
        print('xid is None at:', i)
        continue

      elif xid != None and len(xid) > 1: xid = Table(xid[0])
      
      sp = SDSS.get_spectra(matches=xid)

      df['flux_list'].append(sp[0][1].data['flux'])
      df['wavelength'].append(10. ** sp[0][1].data['loglam'])
      df['z'].append(xid['z'])
      df['ra'].append(xid['ra'])
      df['dec'].append(xid['dec'])
      df['objid'].append(xid['objid'])

    except:
      print('Failed to download at:', i)
      n_errors = n_errors + 1

  df = pd.DataFrame(df)
  print('df.head()', df.head())
  df.to_pickle('data/sdss/spectra_' + str(from_sp) + '-' + str(to_sp) + '.pkl')

  t_end = time.clock()

  t_delta = t_end - t_start
  n_downloads = len(ra) - 1
  print("time for " + str(n_downloads) + " stellar objects:", t_delta)


download_spectra(
  coord_list_url = "data/sdss/coordinate_list.csv",
  from_sp = 5001,
  to_sp = 10000
)

# df = pd.read_pickle('data/sdss/spectra_0-5000.pkl')


