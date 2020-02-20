import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import time as time
import datetime
import math
import seaborn as sn

# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import History, TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Activation, Input, concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical

from spectral_analysis.plotify import Plotify

# This is a mixed input neural network that combines a CNN with an MLP.
# Inputs:
#    - df: pandas dataframe with training data
#    - batch_size: batch size (integer)
#    - hidden_layers: an array of numbers that represent the number of hidden layers and the
#                     number of neurons in each. [128, 128, 128] is 3 hidden layers with 128
#                     neurons each
#    - n_ephoch: the number of epochs the system trains
#
# It then prepares, trains and saves the model to disk so you can load it later. Currently it is 
# a binary classifier, but it can be easily changed
# It also automatically scales the data. This should speed up the process of training


def get_incorrect_predictions(model, X_test, X_test_spectra, y_test, df):
	classes = ['galaxy', 'quasar', 'star']
	predictions = model.predict(X_test).argmax(axis=1)
	y_test = y_test.argmax(axis=1)
	indices = [i for i,v in enumerate(predictions) if predictions[i] != y_test[i]]
	
	wrong_predictions = []
	for i in indices:
		wrong_prediction = {'spectrum': X_test_spectra[i],
							'predicted': classes[predictions[i]],
							'target_class': classes[y_test[predictions[i]]]}
	
		wrong_predictions.append(wrong_prediction)
	
	nth_prediction = 2

	plotify = Plotify()

	spectrum_y = wrong_predictions[nth_prediction]['spectrum']
	spectrum_x = df[['wavelength'][0]][0]

	print('len(spectrum_y', len(spectrum_y))
	print('len(spectrum_x', len(spectrum_x))

	fig, ax = plotify.plot(x=spectrum_x,
						   y=spectrum_y,
						   xlabel='Frequencies',
						   ylabel='Flux',
						   title='title',
  						   figsize=(12, 8),
    					   show_plot=True,
    					   filename=('filename'),
    					   save=False,
    					   color='orange',
    					   ymin=-5,
    					   ymax=12,
    					   xmin=3800,
    					   xmax=9100)

	plt.plot(x=spectrum_x, y=spectrum_y)
	plt.show()

def evaluate_model(model, X_test, y_test):
	classes = ['galaxy', 'quasar', 'star']
	y_pred = model.predict(X_test)
	matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

	df_cm = pd.DataFrame(matrix,
						 index=[i for i in classes],
						 columns=[i for i in classes])

	fig, ax = plt.subplots(figsize=(10,7))
	sn.heatmap(df_cm, annot=True, annot_kws={"size": 15}, fmt='g')
	ax.set_ylabel('Predicted Class', color='black')
	ax.set_xlabel('Target Class', color='black')
	ax.set_title('Confusion Matrix')
	plt.show()
  
def summarize_results():
	print('hello')

def main():
    print('helper_functions main()')

if __name__ == "__main__":
	main()