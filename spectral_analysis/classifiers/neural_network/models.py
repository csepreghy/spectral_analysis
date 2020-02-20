import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import History
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D, Input, concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical

from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
from kerastuner import HyperModel

import time

LOG_DIR = f"{int(time.time())}"

class CNNModel:
    def __init__(self, df_fluxes):
        print('NeuralNetworkModel init')
        self.input_length = len(df_fluxes.columns) - 1
        print(f'self.input_length = {self.input_length}')

    def _prepare_data(self, df_source_info, df_fluxes):
        n_classes = 3
        columns = []

        df_source_info['class'] = pd.Categorical(df_source_info['class'])
        df_dummies = pd.get_dummies(df_source_info['class'], prefix='category')
        df_dummies.columns = ['category_GALAXY', 'category_QSO', 'category_STAR']
        df_source_info = pd.concat([df_source_info, df_dummies], axis=1)

        for column in df_source_info.columns:
            if column not in ['class', 'dec', 'ra', 'plate', 'wavelength', 'objid', 'subClass']:
                columns.append(column)

        X = np.delete(df_fluxes.values, 0, axis=1)
        print(f'X_fluxes = {X}')
        y = []

        print(f'df_source_info = {df_source_info}')

        for index, spectrum in df_source_info[columns].iterrows():
            category_GALAXY = spectrum["category_GALAXY"]
            category_QSO = spectrum["category_QSO"]
            category_STAR = spectrum["category_STAR"]

            y_row = [category_GALAXY, category_QSO, category_STAR]

            y.append(y_row)
    
        print(f'len(X) = {len(X)}')
        print(f'len(y) = {len(y)}')

        return X, y


    def _fit(self, X_train, y_train, X_test, y_test):
        tuner = RandomSearch(self._build_model,
                             objective='val_accuracy',
                             max_trials=10,
                             executions_per_trial=1,
                             directory=LOG_DIR)
    
        tuner.search(x=X_train,
                     y=y_train,
                     epochs=1,
                     batch_size=64,
                     validation_data=(X_test, y_test))

    def _build_model(self, hp):
        model = Sequential()

        model.add(Conv1D(filters=hp.Choice('conv_filters_input_layer', values=[16, 32, 64, 128, 256, 512]),
                         kernel_size=hp.Choice('kernel_size_input_layer', values=[2, 3, 4, 8, 16]),
                         activation='relu',
                         input_shape=(self.input_length, 1)))

        for i in range(hp.Int('n_cnn_layers', 1, 4)):
            model.add(Conv1D(filters=hp.Choice(f'conv_{i}_filters', values=[16, 32, 64, 128, 256, 512]),
                             kernel_size=hp.Choice(f'conv_{i}_filters_kernel_size', values=[2, 3, 4, 8, 16]),
                             activation='relu'))
                                
        # model.add(Dropout(0.5))
        # model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())

        for i in range(hp.Int('n_dense_layers', 1, 4)):
            model.add(Dense(hp.Choice(f'dense_{i}_filters', values=[16, 32, 64, 128, 256, 512]), activation='relu'))

        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def run(self, df_source_info, df_fluxes):
        X, y = self._prepare_data(df_source_info, df_fluxes)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_val = scaler.transform(X_val)

        X_train = np.expand_dims(X_train, axis=2)
        X_test = np.expand_dims(X_test, axis=2)

        y_train = np.array(y_train)
        y_test = np.array(y_test)

        self._fit(X_train, y_train, X_test, y_test)

def build_mlp(input_shape):
  model = Sequential()

  model.add(Dense(256, input_dim=input_shape, activation='relu', kernel_initializer='he_uniform'))
  model.add(Dense(256, input_dim=256, activation='relu', kernel_initializer='he_uniform'))

  return model

def build_models(hp, input_shapes, n_classes):
    cnn = build_cnn(input_length=input_shapes['fluxes'], hp=hp)
    mlp = build_mlp(input_shape=input_shapes['source_info'])
    
    combined = concatenate([cnn.output, mlp.output])

    final_classifier = Dense(128, activation="relu")(combined)
    final_classifier = Dense(n_classes, activation="softmax")(final_classifier)
    
    model = Model(inputs=[mlp.input, cnn.input], outputs=final_classifier)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    
    return model

class HyperSpaceModel(HyperModel):
    def __init__(self, n_classes, input_shape):
        self.n_classes = n_classes
        self.input_shape = input_shape

    def build(self, hp):
        cnn = build_cnn(input_length=self.input_shape, hp=hp)
        

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        return model
