import pandas as pd
from astroquery.nist import Nist
import astropy.units as u
import numpy as np
import pickle


wlist=[5650, 6500, 7000] #wavelength belong to important elemnts


def elemnts_existence(f_list, w_list, wlist):
    i = 0

    while i < len(f_list):

        i =+1
        print(i)
        # j = 0
        # # ind=[]
        # # flux=[]
        # # WL=[]
        # while j < len(df["wavelength"]):
        #     j = j + 1
        #     # k=0
        #     # while k < len(ws):
        #     #     k=k+1
        ws = []
        flux = []
        ind = []
        for wk in wlist:
            # print(wk)
            eps = 10
            lower = wk - eps
            upper = wk + eps
            if (lower < w_list[i] < upper):
                ws.append(w_list[i])
                flux.append(f_list[i])
                ind.append(i)

    df_new = df.drop(columns={"wavelength", "flux"})
    df_new["flux_list"] = flux
    df_new["wavelength"] = ws
    df_new["index"] = ind

    df_new.to_csv("data/sdssFE/FE.csv")

    #check if these flux values are minimum or maximum then add a column as 1 or one for each of them into the dataframe


with open('../data/sdss/999.pkl', 'rb') as f:

    x = pickle.load(f)
df = pd.DataFrame(x)
# print(df.values)

fluxs = pd.DataFrame(df["flux_list"])#.to_dict()
waves= pd.DataFrame(df["wavelength"])#].to_dict()

# .to_list()
for i in range(1): #range(len(waves)):
    list_waves=list(waves.loc[i][0])
    list_flux=list(fluxs.loc[i][0])
    print(i)

    elemnts_existence(list_flux, list_waves, wlist)

    #
    # for i in range(len(list_waves)):
    #     f= list_flux[i]
    #     print(f)
    #     w=list_waves[i]
    #     print(w)


# print(list_waves)
# print(fluxs.loc[0])
# df=pd.read_csv("../data/sdss/999.pkl")
# df.head()


#





