import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import time as time
import datetime
import math

import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import History
from tensorflow.keras.layers import Dense, Dropout, Activation, GlobalAveragePooling1D, Flatten, Conv1D, MaxPooling1D, MaxPooling2D, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical


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

  df['class'] = pd.Categorical(df['class'])
  df_dummies = pd.get_dummies(df['class'], prefix='category')
  df = pd.concat([df, df_dummies], axis=1)
  df_spectra = df[['flux_list']]
  df_source_info = df.drop(columns={'flux_list'})

  for column in df_source_info.columns:
    if column not in ['class', 'dec', 'ra', 'plate', 'wavelength', 'objid', 'subClass']:
      columns.append(column)

  X_source_info = []
  X_spectra = []
  y = []

  for index, spectrum in df_source_info[columns].iterrows():
    X_row = []

    # adding the spectral lines
    
    spectral_lines = spectrum['spectral_lines']

    if type(spectral_lines) == list:
      for spectral_line in spectral_lines:
        X_row.append(spectral_line)

    elif math.isnan(spectral_lines):
      spectral_lines = [0] * 14 # number of spectral lines is 14
  
      for spectral_line in spectral_lines:
          X_row.append(spectral_line)

    X_row.append(spectrum['z'])
    X_row.append(spectrum['zErr'])
    X_row.append(spectrum['petroMag_u'])
    X_row.append(spectrum['petroMag_g'])
    X_row.append(spectrum['petroMag_r'])
    X_row.append(spectrum['petroMag_i'])
    X_row.append(spectrum['petroMag_z'])
    X_row.append(spectrum['petroMagErr_u'])
    X_row.append(spectrum['petroMagErr_g'])
    X_row.append(spectrum['petroMagErr_r'])
    X_row.append(spectrum['petroMagErr_i'])
    X_row.append(spectrum['petroMagErr_z'])

    category_GALAXY = spectrum['category_GALAXY']
    category_QSO = spectrum['category_QSO']
    category_STAR = spectrum['category_STAR']

    y_row = [category_GALAXY, category_QSO, category_STAR]

    X_source_info.append(X_row)
    y.append(y_row)
  
  for _, spectrum in df_spectra.iterrows():
    X_row = []
    
    flux_list = spectrum['flux_list']
    
    for flux in flux_list:
      X_row.append(flux)
    
    X_spectra.append(X_row)


  X_train_source_info, X_test_source_info, y_train, y_test = train_test_split(X_source_info, y, test_size=0.2)
  
  
  X_train_source_info, X_val_source_info, y_train, y_val = train_test_split(X_train_source_info, y_train, test_size=0.2)

  X_train_spectra, X_test_spectra = train_test_split(X_spectra, None, test_size=0.2)
  X_train_spectra, X_val_spectra = train_test_split(X_train_spectra, None, test_size=0.2)

  X_train_source_info = scaler.fit_transform(X_train_source_info)
  X_test_source_info = scaler.transform(X_test_source_info)
  X_val_source_info = scaler.transform(X_val_source_info)

  X_train_spectra = scaler.fit_transform(X_train_spectra)
  X_test_spectra = scaler.transform(X_test_spectra)
  X_val_spectra = scaler.transform(X_val_spectra)

  n_classes = 3

  # model_m = Sequential()
  # model_m.add(Conv1D(100, 10, activation='relu', input_shape=(X_train_spectra.shape[1], 1)))
  # model_m.add(Conv1D(100, 10, activation='relu'))
  # model_m.add(MaxPooling1D(1))
  # model_m.add(Conv1D(160, 10, activation='relu'))
  # model_m.add(Conv1D(160, 10, activation='relu'))
  # model_m.add(GlobalAveragePooling1D())
  # model_m.add(Dropout(0.5))
  # model_m.add(Dense(num_classes, activation='softmax'))
  # print(model_m.summary())
  # model_m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  print('X_test_spectra.shape', type(X_test_spectra))

  X_train_spectra = np.expand_dims(X_train_spectra, axis=2)
  X_test_spectra = np.expand_dims(X_test_spectra, axis=2)

  y_train = np.array(y_train)
  y_test = np.array(y_test)

  print('y_train.shape', y_train.shape)
  print('y_test.shape', type(y_test))

  model_m = Sequential()
  model_m.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_spectra.shape[1], 1)))
  model_m.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
  model_m.add(Dropout(0.5))
  model_m.add(MaxPooling1D(pool_size=2))
  model_m.add(Flatten())
  model_m.add(Dense(100, activation='relu'))
  model_m.add(Dense(n_classes, activation='softmax'))
  model_m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  history = model_m.fit(X_train_spectra, y_train, validation_data=(X_test_spectra, y_test), epochs=60, verbose=0)

  # history = model_m.fit(X_train_spectra,
  #                       y_train,
  #                       batch_size=12,
  #                       epochs=20,
  #                       callbacks=callbacks_list,
  #                       verbose=1)

  # # define model
  # model = Sequential()
  # # model.add(Dense(256, input_dim=X_train_source_info.shape[1], activation='relu', kernel_initializer='he_uniform'))
  # # model.add(Dense(256, input_dim=256, activation='relu', kernel_initializer='he_uniform'))
  # # model.add(Dense(256, input_dim=256, activation='relu', kernel_initializer='he_uniform'))
  # # model.add(Dense(3, activation='softmax'))

  # # # define two sets of inputs
  # input_source_info = Input(shape=(X_train_source_info.shape[1],))
  # input_spectra = Input(shape=(X_train_spectra.shape[1],))
  
  # # # the first branch operates on the first input
  # model_source_info = Dense(8, activation="relu")(input_source_info)
  # model_source_info = Dense(4, activation="relu")(model_source_info)
  # model_source_info = Model(inputs=input_source_info, outputs=model_source_info)

  # print('model_source_info', model_source_info)
  
  # # filters for the CNN
  # filters = (8, 16, 32)

  # for (i, f) in enumerate(filters):
  #   # if this is the first CONV layer then set the input
  #   # appropriately
  #   if i == 0:
  #     model_spectra = input_spectra

  #   # CONV => RELU => BN => POOL
  #   model_spectra = Conv1D(f, (3), padding="same")(model_spectra)
  #   model_spectra = Activation("relu")(model_spectra)
  #   # model_spectra = BatchNormalization(axis=chanDim)(model_spectra)
  #   model_spectra = MaxPooling2D(pool_size=(2, 2))(model_spectra)

  # # the second branch opreates on the second input
  # model_spectra = Dense(64, activation="relu")(input_spectra)
  # model_spectra = Dense(32, activation="relu")(model_spectra)
  # model_spectra = Dense(4, activation="relu")(model_spectra)
  # model_spectra = Model(inputs=input_spectra, outputs=model_spectra)

  # # flatten the volume, then FC => RELU => BN => DROPOUT
  # model_spectra = Flatten()(model_spectra)
  # model_spectra = Dense(16)(model_spectra)
  # model_spectra = Activation("relu")(model_spectra)
  # # model_spectra = BatchNormalization(axis=chanDim)(model_spectra)
  # model_spectra = Dropout(0.5)(model_spectra)

  # # apply another FC layer, this one to match the number of nodes
  # # coming out of the MLP
  # model_spectra = Dense(4)(model_spectra)
  # model_spectra = Activation("relu")(model_spectra)

  # # construct the CNN
  # model = Model(inputs, model_spectra)

  # # combine the output of the two branches
  # combined_input = concatenate([model_source_info.output, model_spectra.output])

  # # apply a FC layer and then a regression prediction on the combined outputs
  # final_classifier = Dense(2, activation="relu")(combined)
  # final_classifier = Dense(3, activation="softmax")(final_classifier)
  
  # # our model will accept the inputs of the two branches and
  # # then output a single value
  # model = Model(inputs=[x.input, y.input], outputs=z)

  # opt = SGD(lr=0.01, momentum=0.9)
  # model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

  # print("[INFO] training model...")
  # start = time.time()

  # y_train = np.array(y_train)
  # y_test = np.array(y_test)
  # print('y_train.shape', y_train.shape)

  # history = model.fit([X_train_source_info, X_train_spectra], y_train, validation_data=([X_test_source_info, X_test_spectra], y_test), epochs=100, verbose=0)


  # evaluate the model
  _, train_acc = model_m.evaluate(X_train_spectra, y_train, verbose=0)
  _, test_acc = model_m.evaluate(X_test_spectra, y_test, verbose=0)
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

  return model_m


def train_test_split(X, y, test_size):
  if y is not None and len(X) != len(y): print('X and y does not have the same length')
  
  n_test = round(len(X) * test_size)
  n_train = len(X) - n_test
  
  X_test = X[-n_test:]
  X_train = X[:n_train]

  # print('type(X_test', type(X_test))

  if y is not None:
    y_test = y[-n_test:]
    y_train = y[:n_train]

  if y is not None:
    return X_train, X_test, y_train, y_test
  
  else:
    return X_train, X_test


# def run_cnn(df, config)