from __future__ import division
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from tqdm.auto import tqdm

def merge_data(filenames):
    with open('data/sdss/spectra-meta/50-100/' + filenames[0], 'rb') as g:
        data = pickle.load(g)

    for i in tqdm(range(len(filenames))):
        with open('data/sdss/spectra-meta/50-100/' + filenames[i], 'rb') as f:
            x = pickle.load(f)
            datalist = pd.concat([data, x], ignore_index=True)
            data = datalist

    return data

def main():
	path = 'data/sdss/spectra-meta/50-100'
	filenames = [f for f in listdir(path) if isfile(join(path, f))]
	try: filenames.remove('.DS_Store')
	except: print('.DS_Store is not in folder')
	
	print(f'filenames = {filenames}')

	all_data = merge_data(filenames)
	print(f'all_data = {all_data}')
	all_data.to_parquet('data/sdss/50-100_merged.parquet')

	# df = pd.read_parquet('data/sdss/spectra-meta/0-50_merged.parquet')
	# print('df parquet', df)

if __name__ == "__main__":
	main()