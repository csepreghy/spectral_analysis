import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import History

history = History()



def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('Truth')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))

  ds = ds.batch(batch_size)

  return ds

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

  # train, test = train_test_split(df, test_size=0.2)
  # train, val = train_test_split(train, test_size=0.2)

  columns = []

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
    for spectral_line in spectral_lines: X_row.append(spectral_line)

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

  print('len(X_train)', len(X_train))

  X_train_std = scaler.fit_transform(X_train)
  X_test_std = scaler.transform(X_test)
  val_X_std = scaler.transform(X_val)



  scaled_train_df = pd.DataFrame(X_train_std)
  scaled_train_df['Truth'] = y_train

  scaled_test_df = pd.DataFrame(X_test_std)
  scaled_test_df['Truth'] = y_test

  scaled_val_df = pd.DataFrame(val_X_std)
  scaled_val_df['Truth'] = y_val

  # df = pd.concat([df, dfDummies], axis=1)

  train_ds = df_to_dataset(scaled_train_df, batch_size=config['batch_size'])
  val_ds = df_to_dataset(scaled_val_df, batch_size=config['batch_size'])
  test_ds = df_to_dataset(scaled_test_df, batch_size=config['batch_size'])

  feature_labels = list(scaled_train_df.columns.values)

  for feature_batch, label_batch in train_ds.take(1):
    print('A batch of Truths:', label_batch)
    feature_labels = list(feature_batch.keys())


  feature_columns = []

  for header in feature_labels:
    feature_columns.append(tf.feature_column.numeric_column(str(header) + 'somecrap'))

  feature_layer = keras.layers.DenseFeatures(feature_columns)

  model = keras.Sequential()
  model.add(feature_layer)

  for n_neurons in config['hidden_layers']:
    model.add(keras.layers.Dense(units=n_neurons, activation='relu'))
  
  model.add(keras.layers.Dense(3, activation='softmax'))

  # custom optimizer
  opt = SGD(lr=0.01, momentum=0.9)

  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  print('X_train_std', X_train_std)
  history = model.fit(X_train_std, y_train, validation_data=(val_X_std, y_val), epochs=config['n_epochs'])

  # model.fit(X_train_std, y_train, epochs=config['n_epochs'])

  _, test_acc = model.evaluate(X_test_std, y_test)

  print('train_acc', train_acc)
  print('test_acc', test_acc)

  # serialize model to JSON
  model_json = model.to_json()
  with open("model.json", "w") as json_file:
      json_file.write(model_json)
  # serialize weights to HDF5
  model.save_weights("model.h5")

  loss, accuracy = model.evaluate(test_ds)
  print("Accuracy", accuracy)

  return model
