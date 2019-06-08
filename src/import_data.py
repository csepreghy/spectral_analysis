import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import time as time
from astropy.table import Table

from astroquery.sdss import SDSS
import astropy.units as u
from astropy import coordinates as coords
from astropy.table import Table

import pickle


def get_save_SDSS_from_coordinates(ra, dec):
    n=0

    df = {}
    df['flux_list'] = []  # ': flux,
    df['wavelength'] = []  #: wavelength,
    df['z'] = []  #: z,
    df['ra'] = []  # ': xid['ra'],
    df['dec'] = []  #: xid['dec'],
    df['objid'] = []  #: xid['objid'],
    #     df['coordinate'] = []  #: coordinate
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
            #
            #         print(xid)
            #         print(counter)
            #         print(type(xid))

            #         typ="<class 'NoneType'>"
            # print("type:",type(xid))
            # print(counter)
            # print(len(xid))
            # print(xid)
            # print(Table(xid[0]))
            if xid != None:

                if len(xid) > 1:
                    # print(len(xid))
                    # print(xid[0])
                    # print(xid)
                    xid = Table(xid[0])
                elif len(xid) == 1:
                    xid = xid
            elif xid == None:
                # print("here")
                number_none = 1+ number_none
                print(counter)
                continue



            # print("this is first row:", xid[0])
            # print(type(xid))

            # print(xid)

            #         print(type(xid))
            #         print(counter)

            sp = SDSS.get_spectra(matches=xid)
            # print(sp)

            # plate_tab = xid["plate"]

            # plate = plate_tab[0]
            # print(plate)
            # print("here")
            # so = SDSS.query_specobj(plate=plate  , fields=['ra', 'dec', 'z', 'zErr'])
            # fSpecClassN=so['fSpecClassN']
            # soclass = SDSS.query_specobj(plate=plate , fields=['specClass'][1])
            # sdf=pd.DataFrame(so)
            # print(soclass)
            # fSpecClassN= soclass['specClass'][1]
            # print(fSpecClassN)


            df['flux_list'].append(sp[0][1].data['flux'])
            df['wavelength'].append(10. ** sp[0][1].data['loglam'])
            df['z'].append(xid['z'])
            df['ra'].append(xid['ra'])
            df['dec'].append(xid['dec'])
            df['objid'].append(xid['objid'])
            # df['plate'].append(plate)
            #         df['coordinate'].append(coordinate)
            # df['zErr'].append(so['zErr'])
            # df['fSpecClassN'].append(so['fSpecClassN'])

            # print(df)

        except:
            print("Error")
            n = n +1
    # print(df)
    df = pd.DataFrame(df)
    df.to_pickle('data/sdss/Nikki_35001-40000.pkl')
    print("number of None values for Xid is:", number_none)
