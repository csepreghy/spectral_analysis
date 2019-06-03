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
    df['zErr'] = []

    counter = -1
    length = len(ra) - 1
    print(length)
    number_none = 0
    while counter < int(length):
# <<<<<<< HEAD
        try:
            counter += 1
            pos = coords.SkyCoord((ra[counter]) * u.deg, (dec[counter]) * u.deg, frame='icrs')
            xid = SDSS.query_region(pos, spectro=True)#, radius=5 * u.arcsec)

            #         print(xid)
            #         print(counter)
            #         print(type(xid))

            #         typ="<class 'NoneType'>"
            # print("type:",type(xid))
            # print(counter)
            # print(len(xid))
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
                number_none = +1
                print(counter)
                continue

            # print("this is first row:", xid[0])
            # print(type(xid))

            # print(xid)

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

        except:
            n = +1

    df = pd.DataFrame(df)
    df.to_pickle('data/sdss/'+str(length)+'.pkl')
    print("number of None values for Xid is:", number_none)

        #         print(counter , "Done")
# # =======
#         counter += 1
#         pos = coords.SkyCoord((ra[counter]) * u.deg, (dec[counter]) * u.deg, frame='icrs')
#         xid = SDSS.query_region(pos, spectro=True)#, radius=5 * u.arcsec)
#
#         #         print(xid)
#         #         print(counter)
#         #         print(type(xid))
#
#         #         typ="<class 'NoneType'>"
#         if xid != None:
#             #             print(len(xid))
#
#             if len(xid) > 1:
#                 xid = Table(xid[0])
#             elif len(xid) == 1:
#                 xid = xid
#         else:
#             number_none =+1
#             print(counter)
#             continue
#
#         #         print(type(xid))
#         #         print(counter)
#
#         sp = SDSS.get_spectra(matches=xid)
#         plate_tab = xid["plate"]
#         plate = plate_tab[0]
#         #         print(plate)
#         so = SDSS.query_specobj(plate=plate, fields=['ra', 'dec', 'z', 'zErr'])
#         #         print("here")
#
#         df['flux_list'].append(sp[0][1].data['flux'])
#         df['wavelength'].append(10. ** sp[0][1].data['loglam'])
#         df['z'].append(xid['z'])
#         df['ra'].append(xid['ra'])
#         df['dec'].append(xid['dec'])
#         df['objid'].append(xid['objid'])
#         #         df['coordinate'].append(coordinate)
#         df['zErr'].append(so['zErr'])
#
#     #         print(counter , "Done")
# # >>>>>>> c39b09ef5d75b2cb6af720d5e37fbb5c1cdcdf69
#

# # <<<<<<< HEAD



# =======
# >>>>>>> c39b09ef5d75b2cb6af720d5e37fbb5c1cdcdf69
