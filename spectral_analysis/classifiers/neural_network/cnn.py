import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import History, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical

from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

import time

from spectral_analysis.data_preprocessing.data_preprocessing import get_joint_classes

LOG_DIR = f"{int(time.time())}"

class CNN:
    def __init__(self, df_fluxes, mainclass=None):
        self.input_length = len(df_fluxes.columns) - 1
        self.mainclass = mainclass

        print(f'self.input_length = {self.input_length}')

    def _prepare_data(self, df_source_info, df_fluxes):
        columns = []

        if self.mainclass == None:
            df_source_info['label'] = [x.decode('utf-8') for x in df_source_info['class']]

        else:
            df_source_info, df_fluxes = get_joint_classes(df_source_info, df_fluxes, self.mainclass)

        df_source_info['class'] = pd.Categorical(df_source_info['class'])
        df_dummies = pd.get_dummies(df_source_info['label'], prefix='label')
        df_source_info = pd.concat([df_source_info, df_dummies], axis=1)

        for column in df_source_info.columns:
            if column not in ['class', 'dec', 'ra', 'plate', 'wavelength', 'objid', 'subClass']:
                columns.append(column)

        X = np.delete(df_fluxes.values, 0, axis=1)
        y = []

        print(f'df_source_info = {df_source_info}')
        label_columns = []
        for column in df_source_info.columns:
            if 'label_' in column: label_columns.append(column)
        
        self.n_labels = len(label_columns)

        for _, spectrum in df_source_info[columns].iterrows():
            y_row = spectrum[label_columns]
            y.append(y_row)
    
        print(f'len(X) = {len(X)}')
        print(f'len(y) = {len(y)}')

        return X, y


    def _fit(self, X_train, y_train, X_test, y_test):
        tuner = RandomSearch(self._build_model,
                             objective='val_accuracy',
                             max_trials=10,
                             executions_per_trial=1,
                             directory='logs/keras-tuner/',
                             project_name='autoencoder')
    
        tuner.search(x=X_train,
                     y=y_train,
                     epochs=20,
                     batch_size=64,
                     validation_data=(X_test, y_test),
                     callbacks=[EarlyStopping('val_accuracy', patience=3)])

    def _build_model(self, hp):
        model = Sequential()

        model.add(Conv1D(filters=hp.Choice('conv_filters_input_layer', values=[16, 32, 64, 128, 256, 512]),
                         kernel_size=hp.Choice('kernel_size_input_layer', values=[3, 5, 7, 9]),
                         activation='relu',
                         input_shape=(self.input_length, 1)))

        for i in range(hp.Int('n_cnn_layers', 1, 4)):
            model.add(Conv1D(filters=hp.Choice(f'conv_{i}_filters', values=[16, 32, 64, 128, 256, 512]),
                             kernel_size=hp.Choice(f'conv_{i}_filters_kernel_size', values=[3, 5, 7, 9]),
                             activation='relu'))
                                
        model.add(Dropout(0.5))
        # model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())

        for i in range(hp.Int('n_dense_layers', 1, 4)):
            model.add(Dense(hp.Choice(f'dense_{i}_filters', values=[16, 32, 64, 128, 256, 512]), activation='relu'))

        model.add(Dense(self.n_labels, activation=hp.Choice('last_activation', values=['softmax', 'tanh'])))
        model.compile(loss='categorical_crossentropy',
                      optimizer=hp.Choice('optimizer', values=['adam', 'nadam']),
                      metrics=['accuracy'])

        return model

    def train(self, df_source_info, df_fluxes):
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

        print(f'X_train = {X_train.shape}')
        print(f'y_train = {y_train.shape}')
        print(f'X_test = {X_test.shape}')
        print(f'y_test = {y_test.shape}')
        self._fit(X_train, y_train, X_test, y_test)

def main():
    df_fluxes = pd.read_hdf('data/sdss/preprocessed/interpolated_1536/0-50_i_fluxes_1536.h5', key='fluxes')
    df_source_info = pd.read_hdf('data/sdss/preprocessed/interpolated_1536/0-50_i_fluxes_1536.h5', key='source_info')
    
    df_fluxes = df_fluxes.head(50)
    df_source_info = df_source_info.head(50)

    cnn = CNN(df_fluxes, mainclass='QSO')
    cnn.train(df_source_info, df_fluxes)

if __name__ == "__main__":
    main()
