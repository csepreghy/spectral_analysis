import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd

from src.import_data import get_gama_dataset_from_query, get_gama_dataset_from_csv, get_SDSS_dataset_from_query
from src.neural_network_classifier import run_neural_network

# df = pd.read_hdf('train.h5')
# model = run_neural_network(df, batch_size=5, hidden_layers=[128, 128], n_epochs=5)

gama_df = get_gama_dataset_from_csv()
print(gama_df.head())

#sp = get_SDSS_dataset_from_query()