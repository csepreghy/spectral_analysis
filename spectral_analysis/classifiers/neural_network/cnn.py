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

LOG_DIR = f"{int(time.time())}"

class CNN:
    def __init__(self, df_fluxes, max_trials, epochs):
        self.input_length = len(df_fluxes.columns) - 1
        self.max_trials = max_trials
        self.epochs = epochs

    def _prepare_data(self, df_source_info, df_fluxes):
        columns = []

        df_source_info['class'] = pd.Categorical(df_source_info['class'])
        df_dummies = pd.get_dummies(df_source_info['class'], prefix='category')
        df_dummies.columns = ['category_GALAXY', 'category_QSO', 'category_STAR']
        df_source_info = pd.concat([df_source_info, df_dummies], axis=1)

        for column in df_source_info.columns:
            if column not in ['class', 'dec', 'ra', 'plate', 'wavelength', 'objid', 'subClass']:
                columns.append(column)

        X = np.delete(df_fluxes.values, 0, axis=1)
        y = []

        print(f'df_source_info = {df_source_info}')

        for _, spectrum in df_source_info[columns].iterrows():
            category_GALAXY = spectrum["category_GALAXY"]
            category_QSO = spectrum["category_QSO"]
            category_STAR = spectrum["category_STAR"]

            y_row = [category_GALAXY, category_QSO, category_STAR]

            y.append(y_row)

        return X, y


    def _fit(self, X_train, y_train, X_test, y_test, X_val, y_val):
        tuner = RandomSearch(self._build_model,
                             objective='val_accuracy',
                             max_trials=self.max_trials,
                             executions_per_trial=1,
                             directory='logs/keras-tuner/',
                             project_name='cnn')

        tuner.search_space_summary()
    
        tuner.search(x=X_train,
                     y=y_train,
                     epochs=self.epochs,
                     batch_size=32,
                     validation_data=(X_val, y_val),
                     callbacks=[EarlyStopping('val_accuracy', patience=4)])
        
        print(tuner.results_summary())
        model = tuner.get_best_models(num_models=1)[0]
        print(model.summary())

        # Evaluate Best Model #
        _, train_acc = model.evaluate(X_train, y_train, verbose=0)
        _, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

    def _build_model(self, hp):
        hyperparameters = {
            'n_conv_layers': hp.Int('n_conv_layers', 1, 4),
            'input_conv_layer_filters': hp.Choice('input_conv_layer_filters', values=[32, 64, 128, 256, 512], default=256),
            'input_conv_layer_kernel_size': hp.Choice('input_conv_layer_kernel_size', values=[3, 5, 7, 9]),
            'n_dense_layers': hp.Int('n_dense_layers', 1, 4),
            'last_activation': hp.Choice('last_activation', ['softmax', 'tanh']),
            'optimizer': hp.Choice('optimizer', values=['adam', 'nadam', 'rmsprop'])
        }
        
        for i in range(hyperparameters['n_conv_layers']):
            hyperparameters[f'conv_layer_{i}_filters'] = hp.Choice(f'conv_layer_{i}_filters',
                                                                   values=[32, 64, 128, 256, 512],
                                                                   default=256)
            hyperparameters[f'conv_layer_{i}_kernel_size'] = hp.Choice(f'conv_layer_{i}_kernel_size',
                                                                       values=[3, 5, 7, 9])
        for i in range(hyperparameters['n_dense_layers']):
            hyperparameters[f'dense_layer_{i}_nodes'] = hp.Choice(f'dense_layer_{i}_nodes',
                                                                   values=[32, 64, 128, 256, 512],
                                                                   default=256)

               
        model = Sequential()

        model.add(Conv1D(filters=hyperparameters['input_conv_layer_filters'],
                         kernel_size=hyperparameters['input_conv_layer_kernel_size'],
                         activation='relu',
                         input_shape=(self.input_length, 1)))

        for i in range(hyperparameters['n_conv_layers']):
            model.add(Conv1D(filters=hyperparameters[f'conv_layer_{i}_filters'],
                             kernel_size=hyperparameters[f'conv_layer_{i}_kernel_size'],
                             activation='relu'))

            model.add(MaxPooling1D(pool_size=hp.Int('max_pool_size', 1, 4)))
        
        model.add(Flatten())

        for i in range(hyperparameters['n_dense_layers']):
            model.add(Dense(hyperparameters[f'dense_layer_{i}_nodes']))

        model.add(Dense(3, activation=hyperparameters['last_activation']))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

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
        X_val = np.expand_dims(X_val, axis=2)

        y_train = np.array(y_train)
        y_test = np.array(y_test)
        y_val = np.array(y_val)

        self._fit(X_train, y_train, X_test, y_test, X_val, y_val)

def main():
    df_fluxes = pd.read_hdf('data/sdss/preprocessed/balanced_spectral_lines.h5', key='fluxes')
    df_source_info = pd.read_hdf('data/sdss/preprocessed/balanced_spectral_lines.h5', key='source_info')
    
    df_fluxes = df_fluxes.head(40)
    df_source_info = df_source_info.head(40)

    cnn = CNN(df_fluxes, max_trials=1, epochs=2)
    cnn.run(df_source_info, df_fluxes)

if __name__ == "__main__":
    main()
