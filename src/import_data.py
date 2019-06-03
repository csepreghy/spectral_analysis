import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import time as time
from astropy.table import Table

from astroquery.sdss import SDSS
import astropy.units as u
from astropy import coordinates as coords

import pickle


def get_save_SDSS_from_coordinates(ra, dec):
    df = {}
    df['flux_list'] = []  # ': flux,
    df['wavelength'] = []  #: wavelength,
    df['z'] = []  #: z,
    df['ra'] = []  # ': xid['ra'],
    df['dec'] = []  #: xid['dec'],
    df['objid'] = []  #: xid['objid'],
    #     df['coordinate'] = []  #: coordinate
    df['zErr'] = []

    counter = -1
    length = len(ra) - 1
    print(length)
    number_none = 0
    while counter < int(length):
        counter += 1
        pos = coords.SkyCoord((ra[counter]) * u.deg, (dec[counter]) * u.deg, frame='icrs')
        xid = SDSS.query_region(pos, spectro=True)#, radius=5 * u.arcsec)

        #         print(xid)
        #         print(counter)
        #         print(type(xid))

        #         typ="<class 'NoneType'>"
        if xid != None:
            #             print(len(xid))

            if len(xid) > 1:
                xid = Table(xid[0])
            elif len(xid) == 1:
                xid = xid
        else:
            number_none =+1
            print(counter)
            continue

        #         print(type(xid))
        #         print(counter)

        sp = SDSS.get_spectra(matches=xid)
        plate_tab = xid["plate"]
        plate = plate_tab[0]
        #         print(plate)
        so = SDSS.query_specobj(plate=plate, fields=['ra', 'dec', 'z', 'zErr'])
        #         print("here")

        df['flux_list'].append(sp[0][1].data['flux'])
        df['wavelength'].append(10. ** sp[0][1].data['loglam'])
        df['z'].append(xid['z'])
        df['ra'].append(xid['ra'])
        df['dec'].append(xid['dec'])
        df['objid'].append(xid['objid'])
        #         df['coordinate'].append(coordinate)
        df['zErr'].append(so['zErr'])

    #         print(counter , "Done")

    df = pd.DataFrame(df)
    df.to_pickle('data/sdss/'+str(length)+'.pkl')
    print("number of None values for Xid is:", number_none)
