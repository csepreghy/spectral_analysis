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



def get_save_SDSS_from_coordinates(ra, dec):
    n = 0

    df = {}
    df['flux_list'] = []  # ': flux,
    df['wavelength'] = []  #: wavelength,
    df['z'] = []  #: z,
    df['ra'] = []  # ': xid['ra'],
    df['dec'] = []  #: xid['dec'],
    df['objid'] = []  #: xid['objid'],
    # df['zErr'] = []
    # df['plate']=[]
    # df['fSpecClassN']=[]

    counter = -1
    length = len(ra) - 1
    print(length)
    number_none = 0
    while counter < int(length):
      counter += 1

      try:
        pos = coords.SkyCoord((ra[counter]) * u.deg, (dec[counter]) * u.deg, frame='icrs')
        xid = SDSS.query_region(pos, spectro=True)#, radius=5 * u.arcsec)

        if xid != None:
          if len(xid) > 1: xid = Table(xid[0])

          elif len(xid) == 1: xid = xid
    
        elif xid == None:
          number_none = number_none + 1
          print('Failed to download at: ' counter)
          continue

        sp = SDSS.get_spectra(matches=xid)

        df['flux_list'].append(sp[0][1].data['flux'])
        df['wavelength'].append(10. ** sp[0][1].data['loglam'])
        df['z'].append(xid['z'])
        df['ra'].append(xid['ra'])
        df['dec'].append(xid['dec'])
        df['objid'].append(xid['objid'])
        # df['plate'].append(plate)
        # df['zErr'].append(so['zErr'])
        # df['fSpecClassN'].append(so['fSpecClassN'])

        # print(df)

      except:
        print("Error")
        n = n + 1

    df = pd.DataFrame(df)
    print('df.head()', df.head())
    df.to_pickle('data/sdss/Nikki_35001-40000.pkl')

coord_list = pd.read_csv("data/sdss/coordinate_list.csv")
start = time.time()

ra_list = coord_list["ra"].tolist()
dec_list = coord_list["dec"].tolist()
end = time.time()
tt = end - start
print("time for listing is:", tt)

start1 = time.clock()
ra = ra_list[40001:40005]
dec = dec_list[40001:40005]
get_save_SDSS_from_coordinates(ra, dec)
end1 = time.clock()

tt1 = end1 - start1
n_downloads = len(ra) - 1
print("time for " + str(n_downloads) + " stellar objects:", tt1)

