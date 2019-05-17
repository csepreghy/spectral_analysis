import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd

from src.neural_network_classifier import run_neural_network

df = pd.read_hdf('train.h5')

model = run_neural_network(df, batch_size=5, hidden_layers=[128, 128], n_epochs=5)
print('model', model)
