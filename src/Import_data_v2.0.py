from astroquery.gama import GAMA
from astroquery.sdss import SDSS
import astropy.units as u
from astropy import coordinates as coords

import pickle

from astropy import coordinates as coords

import pandas as pd
import numpy as np


# def get_gama_dataset_from_query():
#   gama_all_data = GAMA.query_sql('SELECT * FROM AATSpecAll AS s WHERE s.z BETWEEN 0.000000001 AND 10')
#   gama_all_data_df = pd.DataFrame(np.array(gama_all_data)) # convert table to CSV
#   gama_all_data_df.to_csv('data/GAMA.csv') # write csv into data folder

#   return gama_all_data_df


# def get_gama_dataset_from_csv():
#   gama_df = pd.read_csv('data/GAMA.csv')

#   return gama_df

# def get_sample_SDSS_dataset_from_query():
#   pos = coords.SkyCoord('120.01976d +45.916684d', frame='icrs')
#   xid = SDSS.query_region(pos, spectro=True, radius=200*u.arcsec)
#   sp = SDSS.get_spectra(matches=xid)
#   #im = SDSS.get_images(matches=xid, band='g')

#   return sp, xid

def get_save_SDSS_from_coordinates(coord_list):

    df = {}
    df['flux_list'] = []  # ': flux,
    df['wavelength'] = []  #: wavelength,
    df['z'] = []  #: z,
    df['ra'] = []  # ': xid['ra'],
    df['dec'] = []  #: xid['dec'],
    df['objid'] = []  #: xid['objid'],
    df['coordinate'] = []  #: coordinate

    for coordinate in coord_list:
        pos = coords.SkyCoord(coordinate, frame='icrs')
        xid = SDSS.query_region(pos, spectro=True, radius=5 * u.arcsec)
        sp = SDSS.get_spectra(matches=xid)

        df['flux_list'].append(sp[0][1].data['flux'])
        df['wavelength'].append(10. ** sp[0][1].data['loglam'])
        df['z'].append(xid['z'])
        df['ra'].append(xid['ra'])
        df['dec'].append(xid['dec'])
        df['objid'].append(xid['objid'])
        df['coordinate'].append(coordinate)

    df = pd.DataFrame(df)
    df.to_pickle('data/sdss_csv/' + coordinate + '.pkl')
