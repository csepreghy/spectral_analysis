from __future__ import division
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

def merge_data(filenames):
    with open('data/'+filenames[0]+'.pkl', 'rb') as g:
                data = pickle.load(g)
    for i in range(1,len(filenames)):
        with open('data/'+filenames[i]+'.pkl', 'rb') as f:
            x = pickle.load(f)
        datalist = pd.concat([data, x], ignore_index=True)
        data = datalist
    return data

names = ['FinalTable_Nikki(0-10000)', 'FinalTable_10-15-Zoe', 'FinalTable_15-30_Zoe', 'FinalTable_30001-35000(Nikki)', 'FinalTable_35001-40000(Nikki)', 'FinalTable_Andrew(40-45)']
all_data = merge_data(names)
all_data.to_pickle("./data/alldatamerged.pkl")