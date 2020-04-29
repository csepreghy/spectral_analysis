import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import time as time
import datetime
import math
import random
from tqdm.auto import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import History, TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D, Input, concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical

from spectral_analysis.data_preprocessing.data_preprocessing import (remove_bytes_from_class,
                                                                     get_joint_classes,
                                                                     plot_spectrum,
                                                                     interpolate_and_reduce_to)
from spectral_analysis.plotify import Plotify
from spectral_analysis.classifiers.neural_network.helper_functions import train_test_split, evaluate_model, unison_shuffled_copies

class MixedInputModel():
    def __init__(self, mainclass=None):
        self.mainclass = mainclass

    def _prepare_data(self, df_source_info, df_fluxes):
        columns = []
        if self.mainclass == None:
            try: df_source_info['label'] = [x.decode('utf-8') for x in df_source_info['class']]
            except: df_source_info['label'] = df_source_info['class']

        else:
            df_source_info, df_fluxes = get_joint_classes(df_source_info, df_fluxes, self.mainclass)
        
        df_source_info['label'] = pd.Categorical(df_source_info['label'])
        
        df_dummies = pd.get_dummies(df_source_info['label'], prefix='label')
        df_source_info = pd.concat([df_source_info, df_dummies], axis=1)

        for column in df_source_info.columns:
            if column not in ['class', 'dec', 'ra', 'plate', 'wavelength', 'objid', 'subClass', 'label']:
                columns.append(column)

        X_source_info = []
        X_fluxes = np.delete(df_fluxes.values, 0, axis=1)
        y = []

        label_columns = []
        for column in df_source_info.columns:
            if 'label_' in column: label_columns.append(column)
        
        self.n_labels = len(label_columns)
        
        print(f'label_columns = {len(label_columns)}')

        for _, spectrum in tqdm(df_source_info[columns].iterrows(), total=len(df_source_info), desc="Preparing Data: "):
            X_row = []

            for i in range(14):
                column = f'spectral_line_{i}'
                spectral_line = spectrum[column]
                
                if np.isnan(spectral_line):
                    X_row.append(-99)
                
                else:
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

            y_row = spectrum[label_columns]

            if np.isnan(np.sum(X_row)):
              raise Exception(f'Found ya! Row: {X_row}')

            X_source_info.append(X_row)
            y.append(y_row)
        
        array_sum = np.sum(X_source_info)
        array_has_nan = np.isnan(array_sum)

        print('array_has_nan', array_has_nan)

        X_source_info, X_fluxes = unison_shuffled_copies(X_source_info, X_fluxes)
        

        return X_source_info, X_fluxes, y

    def _build_cnn(self, input_length):
        model = Sequential()

        model.add(Conv1D(filters=256, kernel_size=5, activation='relu', input_shape=(input_length, 1)))
        model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        model.add(Dropout(0.2))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))

        return model

    def _build_mlp(self, input_shape):
        model = Sequential()

        model.add(Dense(1024, input_dim=input_shape, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(512, input_dim=input_shape, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(256, input_dim=512, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(128, input_dim=256, activation='relu', kernel_initializer='he_uniform'))

        return model

    def _build_models(self, input_shapes, n_classes):
        cnn = self._build_cnn(input_length=input_shapes['fluxes'])
        mlp = self._build_mlp(input_shape=input_shapes['source_info'])
        
        combined = concatenate([cnn.output, mlp.output])

        final_classifier = Dense(128, activation="relu")(combined)
        final_classifier = Dense(n_classes, activation="sigmoid")(final_classifier)
        
        model = Model(inputs=[mlp.input, cnn.input], outputs=final_classifier)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
        return model
    
    def train(self, df_source_info, df_fluxes):
        X_source_info, X_fluxes, y = self._prepare_data(df_source_info, df_fluxes)

        X_train_source_info, X_test_source_info, y_train, y_test = train_test_split(X=X_source_info, y=y, test_size=0.2)
        X_train_source_info, X_val_source_info, y_train, y_val = train_test_split(X=X_train_source_info, y=y_train, test_size=0.2)

        print(f'X_fluxes = {X_fluxes}')

        X_train_spectra, X_test_spectra = train_test_split(X=X_fluxes, y=None, test_size=0.2)
        X_train_spectra, X_val_spectra = train_test_split(X=X_train_spectra, y=None, test_size=0.2)

        scaler = StandardScaler()

        X_train_source_info = scaler.fit_transform(X_train_source_info)
        X_test_source_info = scaler.transform(X_test_source_info)
        X_val_source_info = scaler.transform(X_val_source_info)

        X_train_spectra = scaler.fit_transform(X_train_spectra)
        X_test_spectra_std = scaler.transform(X_test_spectra)
        X_val_spectra = scaler.transform(X_val_spectra)

        X_train_spectra = np.expand_dims(X_train_spectra, axis=2)
        X_test_spectra_std = np.expand_dims(X_test_spectra_std, axis=2)

        y_train = np.array(y_train)
        y_test = np.array(y_test)

        input_shapes = {'fluxes': X_train_spectra.shape[1], 
                        'source_info': X_train_source_info.shape[1]}

        
        model = self._build_models(input_shapes=input_shapes, n_classes=self.n_labels)

        tensorboard = TensorBoard(log_dir='logs/{}'.format('cnn-mlp_{}'.format(time.time())))
        earlystopping = EarlyStopping(monitor='val_accuracy', patience=13)
        modelcheckpoint = ModelCheckpoint(filepath='best_model_epoch.{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', save_best_only=True),

        callbacks_list = [# modelcheckpoint,
                        earlystopping,
                        tensorboard]

        history = model.fit(x=[X_train_source_info, X_train_spectra],
                            y=y_train,
                            validation_data=([X_test_source_info, X_test_spectra_std], y_test),
                            epochs=24,
                            callbacks=callbacks_list)


        # evaluate the model
        _, train_acc = model.evaluate([X_train_source_info, X_train_spectra], y_train, verbose=0)
        _, test_acc = model.evaluate([X_test_source_info, X_test_spectra_std], y_test, verbose=0)

        print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

        # get_incorrect_predictions(X_test=[X_test_source_info, X_test_spectra_std],
        #                           X_test_spectra=X_test_spectra,
       #                            model=model,
       #                            y_test=y_test,
       #                            df=df)

        evaluate_model(model=model,
                       X_test=[X_test_source_info, X_test_spectra_std],
                       y_test=y_test)

        return model

def main():
    df_fluxes = pd.read_hdf('data/sdss/preprocessed/balanced_spectral_lines.h5', key='fluxes').head(75000)
    df_source_info = pd.read_hdf('data/sdss/preprocessed/balanced_spectral_lines.h5', key='spectral_data').head(75000)
    df_wavelengths = pd.read_hdf('data/sdss/preprocessed/balanced_spectral_lines.h5', key='wavelengths')


    print(f'len(df_fluxes1) = {df_fluxes}')
    print(f'len(df_source_info2) = {df_source_info}')

    mixed_input_model = MixedInputModel()
    mixed_input_model.train(df_source_info, df_fluxes)



if __name__ == "__main__":
    main()