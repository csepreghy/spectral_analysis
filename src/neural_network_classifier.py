import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import time as time
import datetime

import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import History
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

history = History()
style.use('fivethirtyeight')

# This is a simple feed forward neural network that uses Keras and Tensorflow.
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


def run_neural_network(df, config):
  scaler = StandardScaler()

  columns = []


  df = pd.read_pickle('COMPLETE_df.pkl')

  df['class'] = pd.Categorical(df['class'])
  dfDummies = pd.get_dummies(df['class'], prefix='category')
  df = pd.concat([df, dfDummies], axis=1)

  for column in df.columns:
    if column not in ['class', 'dec', 'ra', 'plate', 'wavelength', 'objid']:
      columns.append(column)

  print('columns', columns)

  X = []
  y = []

  for index, spectrum in df[columns].iterrows():
    X_row = []

    spectral_lines = spectrum['spectral_lines']
    for spectral_line in spectral_lines:
      X_row.append(spectral_line)

    flux_list = spectrum['flux_list']
    for flux in flux_list:
      X_row.append(flux)

    X_row.append(spectrum['z'])
    X_row.append(spectrum['zErr'])

    category_GALAXY = spectrum['category_GALAXY']
    category_QSO = spectrum['category_QSO']
    category_STAR = spectrum['category_STAR']

    y_row = [category_GALAXY, category_QSO, category_STAR]

    X.append(X_row)
    y.append(y_row)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


  X_train_std = scaler.fit_transform(X_train)
  X_test_std = scaler.transform(X_test)
  X_val_std = scaler.transform(X_val)

  # define model
  model = Sequential()
  model.add(Dense(256, input_dim=X_train_std.shape[1], activation='relu', kernel_initializer='he_uniform'))
  model.add(Dense(256, input_dim=256, activation='relu', kernel_initializer='he_uniform'))
  model.add(Dense(256, input_dim=256, activation='relu', kernel_initializer='he_uniform'))
  model.add(Dense(3, activation='softmax'))

  opt = SGD(lr=0.01, momentum=0.9)
  model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

  start = time.time()



  y_train = np.array(y_train)
  y_test = np.array(y_test)
  print('y_train.shape', y_train.shape)
  print('type(y_train)', type(y_train))

  history = model.fit(X_train_std, y_train, validation_data=(X_test_std, y_test), epochs=100, verbose=0)
  end = time.time()
  deltaticks = end - start
  converted_ticks = datetime.datetime.now() + datetime.timedelta(microseconds=deltaticks/10)
  print('converted_ticks', converted_ticks)

  # evaluate the model
  _, train_acc = model.evaluate(X_train_std, y_train, verbose=0)
  _, test_acc = model.evaluate(X_test_std, y_test, verbose=0)
  print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
  # plot loss during training
  plt.subplot(211)
  plt.title('Loss')
  plt.plot(history.history['loss'], label='train')
  plt.plot(history.history['val_loss'], label='test')
  plt.legend()
  # plot accuracy during training
  plt.subplot(212)
  plt.title('Accuracy')

  plt.plot(history.history['accuracy'], label='train')
  plt.plot(history.history['val_accuracy'], label='test')
  plt.legend()
  plt.show()

  return model
