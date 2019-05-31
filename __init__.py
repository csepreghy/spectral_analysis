import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd

from astropy.io.fits import HDUList
from astropy import coordinates as coords
import astropy.units as u
from astroquery.sdss import SDSS

from src.Import_data_v2_0 import get_save_SDSS_from_coordinates
from src.neural_network_classifier import run_neural_network

# df = pd.read_hdf('train.h5')
# model = run_neural_network(df, batch_size=5, hidden_layers=[128, 128], n_epochs=5)

# gama_df = get_gama_dataset_from_csv()
# sp, xid = get_sample_SDSS_dataset_from_query()


# coord_list = ['0h8m05.63s +14d50m23.3s', '0h9m06.63s +15d55m23.3s']
with open('data/sdss_coordinates_lowz.txt') as text_file:
  coord_list = text_file.read().splitlines()
  # mystring.replace('\n', ' ').replace('\r', '')

get_save_SDSS_from_coordinates(coord_list[0:5000])
