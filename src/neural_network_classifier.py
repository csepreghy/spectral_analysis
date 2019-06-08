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

  train, test = train_test_split(df, test_size=0.2)
  train, val = train_test_split(train, test_size=0.2)

  columns = [column for column in train.columns if column not in ['Truth']]
  
  train_X_std = scaler.fit_transform(train[columns])
  test_X_std = scaler.transform(test[columns])
  val_X_std = scaler.transform(val[columns])

  print('train.columns', train.columns)
  print('test.columns', test.columns)

  scaled_train_df = pd.DataFrame(train_X_std, index=train.index, columns=columns)
  scaled_test_df = pd.DataFrame(test_X_std, index=test.index, columns=columns)
  scaled_val_df = pd.DataFrame(val_X_std, index=val.index, columns=columns)

  scaled_train_df['Truth'] = train['Truth']
  scaled_test_df['Truth'] = test['Truth']
  scaled_val_df['Truth'] = val['Truth']

  train_ds = df_to_dataset(scaled_train_df, batch_size=config['batch_size'])
  val_ds = df_to_dataset(scaled_val_df, batch_size=config['batch_size'])
  test_ds = df_to_dataset(scaled_test_df, batch_size=config['batch_size'])

  feature_labels = list(df.columns.values)
  print('feature_labels', feature_labels)

  for feature_batch, label_batch in train_ds.take(1):
    print('A batch of p_charges:', feature_batch['p_numberOfInnermostPixelHits'])
    print('A batch of Truths:', label_batch)
    feature_labels = list(feature_batch.keys())


  feature_columns = []

  for header in feature_labels:
    feature_columns.append(tf.feature_column.numeric_column(header))


  feature_layer = keras.layers.DenseFeatures(feature_columns)
  print('feature_layer', feature_layer)

  model = keras.Sequential()
  model.add(feature_layer)

  for n_neurons in config['hidden_layers']:
    model.add(keras.layers.Dense(units=n_neurons, activation='relu'))
  
  model.add(keras.layers.Dense(1, activation='softmax'))

  # custom optimizer
  opt = SGD(lr=0.01, momentum=0.9)

  model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

  model.fit(train_ds, validation_data=val_ds, epochs=config['n_epochs'])

  # serialize model to JSON
  model_json = model.to_json()
  with open("model.json", "w") as json_file:
      json_file.write(model_json)
  # serialize weights to HDF5
  model.save_weights("model.h5")

  loss, accuracy = model.evaluate(test_ds)
  print("Accuracy", accuracy)

  return model
