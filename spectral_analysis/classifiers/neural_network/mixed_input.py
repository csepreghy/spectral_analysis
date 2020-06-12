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

from spectral_analysis.spectral_analysis.data_preprocessing.data_preprocessing import remove_bytes_from_class, get_fluxes_from_h5, get_joint_classes, apply_gaussian_filter
from spectral_analysis.spectral_analysis.plotify import Plotify
from spectral_analysis.spectral_analysis.classifiers.neural_network.helper_functions import train_test_split, evaluate_model, shuffle_in_unison, get_incorrect_predictions

class MixedInputModel():
    def __init__(self, mainclass='NONE', spectral_lines=False, df_wavelengths=None, gaussian=False, epochs=2, load_model=False, model_path=None):
        self.mainclass = mainclass
        self.spectral_lines = spectral_lines
        self.df_wavelengths = df_wavelengths
        self.gaussian = gaussian
        self.epochs = epochs
        self.load_model = load_model
        self.model_path = model_path

    def _prepare_data(self, df_source_info, df_fluxes):
        self.df_source_info = df_source_info

        quasars = df_source_info.loc[df_source_info['class'] == 'QSO']
        print(f'len(quasars) = {len(quasars)}')

        galaxies = df_source_info.loc[df_source_info['class'] == 'GALAXY']
        print(f'len(galaxies) = {len(galaxies)}')

        stars = df_source_info.loc[df_source_info['class'] == 'STAR']
        print(f'len(stars) = {len(stars)}')
        
        if self.mainclass == 'NONE':
            try: df_source_info['label'] = [x.decode('utf-8') for x in df_source_info['class']]
            except: df_source_info['label'] = df_source_info['class']

        else:
            df_source_info, df_fluxes = get_joint_classes(df_source_info, df_fluxes, self.mainclass)
        
        df_source_info['label'] = pd.Categorical(df_source_info['label'])
        
        df_dummies = pd.get_dummies(df_source_info['label'], prefix='label')
        df_source_info = pd.concat([df_source_info, df_dummies], axis=1)

        columns = []
        for column in df_source_info.columns:
            if column not in ['class', 'dec', 'ra', 'plate', 'wavelength', 'objid', 'subClass', 'label']:
                columns.append(column)

        X_source_info = []
        X_fluxes = np.delete(df_fluxes.values, 0, axis=1)
        self.raw_X_fluxes = X_fluxes
        if self.gaussian == True:
            X_fluxes_gaussian = []
            for X_flux in X_fluxes:
                X_flux_gaussian = apply_gaussian_filter(X_flux, sigma=4)
                X_fluxes_gaussian.append(X_flux_gaussian)
            
            X_fluxes = X_fluxes_gaussian
        
        y = []

        if self.mainclass is not None:
            self.label_columns = []
            for column in df_source_info.columns:
                if 'label_' in column: self.label_columns.append(column)
            
            self.n_labels = len(self.label_columns)
            print(f'self.label_columns = {self.label_columns}')

        else: self.n_labels = 3

        for _, spectrum in tqdm(df_source_info[columns].iterrows(), total=len(df_source_info), desc="Preparing Data: "):
            X_row = []
            
            if self.spectral_lines:
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

            if self.mainclass == 'NONE':
                label_GALAXY = spectrum['label_GALAXY']
                label_QSO = spectrum['label_QSO']
                label_STAR = spectrum['label_STAR']

                y_row = [label_GALAXY, label_QSO, label_STAR]

            else: y_row = spectrum[self.label_columns]

            if np.isnan(np.sum(X_row)):
                raise Exception(f'Found ya! Row: {X_row}')

            X_source_info.append(X_row)
            y.append(y_row)
        
        array_sum = np.sum(X_source_info)
        array_has_nan = np.isnan(array_sum)

        indeces = list(range(len(X_source_info)))
        X_source_info = np.array(X_source_info)
        X_fluxes = np.array(X_fluxes)
        
        print(f'X_source_info = {X_source_info}')
        print(f'X_fluxes.shape = {X_fluxes.shape}')
        
        X_source_info, X_fluxes, y, indeces = shuffle_in_unison(X_source_info, X_fluxes, y, indeces)

        return X_source_info, X_fluxes, y, indeces

    def _build_cnn(self, input_length):
        model = Sequential()

        model.add(Conv1D(filters=256, kernel_size=5, activation='relu', input_shape=(input_length, 1)))
        model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
        # model.add(Conv1D(filters=128, kernel_size=5, activation='relu', input_shape=(input_length, 1)))
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))

        return model

    def _build_mlp(self, input_shape):
        model = Sequential()

        # model.add(Dense(256, input_dim=input_shape, activation='relu', kernel_initializer='he_uniform'))
        # model.add(Dropout(0.5))
        model.add(Dense(128, input_dim=input_shape, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dropout(0.5))
        model.add(Dense(64, input_dim=158, activation='relu', kernel_initializer='he_uniform'))

        return model

    def _build_models(self, input_shapes, n_classes):
        cnn = self._build_cnn(input_length=input_shapes['fluxes'])
        mlp = self._build_mlp(input_shape=input_shapes['source_info'])
        
        combined = concatenate([cnn.output, mlp.output])

        final_classifier = Dense(128, activation="relu")(combined)
        final_classifier = Dense(n_classes, activation="softmax")(final_classifier)

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        model = Model(inputs=[mlp.input, cnn.input], outputs=final_classifier)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
        return model
    
    def train(self, df_source_info, df_fluxes, df_wavelengths):
        X_source_info, X_fluxes, y, indeces = self._prepare_data(df_source_info, df_fluxes)
        
        X_train_source_info, X_test_source_info, y_train, y_test, self.i_train, self.i_test = train_test_split(X=X_source_info, y=y, test_size=0.2, indeces=indeces)
        X_train_source_info, X_val_source_info, y_train, y_val, self.i_train, self.i_val = train_test_split(X=X_train_source_info, y=y_train, test_size=0.2, indeces=self.i_train)

        X_train_fluxes, X_test_fluxes = train_test_split(X=X_fluxes, y=None, test_size=0.2)
        X_train_fluxes, X_val_fluxes = train_test_split(X=X_train_fluxes, y=None, test_size=0.2)
        
        # To get the same train test split for raw spectra after gaussian smoothing
        raw_X_train_fluxes, raw_X_test_fluxes = train_test_split(X=self.raw_X_fluxes, y=None, test_size=0.2)
        raw_X_train_fluxes, raw_X_val_fluxes = train_test_split(X=raw_X_train_fluxes, y=None, test_size=0.2)
        
        scaler_source_info = StandardScaler()

        # print(f'X_train_source_info = {X_train_source_info}')
        # print(f'X_train_fluxes = {X_train_fluxes}')

        X_train_source_info_std = scaler_source_info.fit_transform(X_train_source_info)
        X_test_source_info_std = scaler_source_info.transform(X_test_source_info)
        X_val_source_info_std = scaler_source_info.transform(X_val_source_info)

        scaler_fluxes = StandardScaler()

        X_train_fluxes_std = scaler_fluxes.fit_transform(X_train_fluxes)
        X_test_fluxes_std = scaler_fluxes.transform(X_test_fluxes)
        X_val_fluxes = scaler_fluxes.transform(X_val_fluxes)

        X_train_fluxes_std = np.expand_dims(X_train_fluxes_std, axis=2)
        X_test_fluxes_std = np.expand_dims(X_test_fluxes_std, axis=2)

        y_train = np.array(y_train)
        y_test = np.array(y_test)
        y_val = np.array(y_val)

        df_source_info_test = df_source_info.iloc[self.i_test]

        input_shapes = {'fluxes': X_train_fluxes.shape[1], 
                        'source_info': X_train_source_info_std.shape[1]}

        model = self._build_models(input_shapes=input_shapes, n_classes=self.n_labels)
        print(model.summary())


        if self.load_model == True:
            model.load_weights(self.model_path)
            _, train_acc = model.evaluate([X_train_source_info_std, X_train_fluxes_std], y_train, verbose=0)
            _, test_acc = model.evaluate([X_test_source_info_std, X_test_fluxes_std], y_test, verbose=0)
            print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

            evaluate_model(model=model,
                           X_test=[X_test_source_info_std, X_test_fluxes_std],
                           y_test=y_test,
                           df_source_info=self.df_source_info,
                           indeces=self.i_test,
                           classes=self.label_columns)
                           
            # get_incorrect_predictions(model=model,
            #                           X_test_fluxes=[X_test_source_info_std, X_test_fluxes_std],
            #                           X_test_spectra=X_test_fluxes,
            #                           raw_X_test_spectra=raw_X_test_fluxes,
            #                           y_test=y_test,
            #                           df_source_info_test=df_source_info_test,
            #                           df_wavelengths=df_wavelengths,
            #                           gaussian=self.gaussian)
  
        
        elif self.load_model == False:
            tensorboard = TensorBoard(log_dir='logs/{}'.format('mixed-input{}'.format(time.time())))
            earlystopping = EarlyStopping(monitor='val_accuracy', patience=50)
            modelcheckpoint = ModelCheckpoint(filepath='logs/mixed_input_32k-50epoch.epoch{epoch:02d}-val_loss_{val_loss:.2f}.h5',
                                              monitor='val_loss',
                                              save_best_only=True)

            callbacks_list = [modelcheckpoint,
                              # earlystopping,
                              tensorboard]

            history = model.fit(x=[X_train_source_info_std, X_train_fluxes_std],
                                y=y_train,
                                validation_data=([X_val_source_info_std, X_val_fluxes], y_val),
                                epochs=self.epochs,
                                batch_size=32,
                                callbacks=callbacks_list)

            # evaluate the model
            _, train_acc = model.evaluate([X_train_source_info_std, X_train_fluxes_std], y_train)
            _, test_acc = model.evaluate([X_test_source_info_std, X_test_fluxes_std], y_test)

            print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
            
            evaluate_model(model=model,
                           X_test=[X_test_source_info_std, X_test_fluxes_std],
                           y_test=y_test,
                           df_source_info=self.df_source_info,
                           indeces=self.i_test,
                           classes=self.label_columns)

            # get_incorrect_predictions(model=model,
            #                           X_test_fluxes=[X_test_source_info_std, X_test_fluxes_std],
            #                           X_test_spectra=X_test_fluxes,
            #                           raw_X_test_spectra=raw_X_test_fluxes,
            #                           y_test=y_test,
            #                           df_source_info_test=df_source_info_test,
            #                           df_wavelengths=df_wavelengths,
            #                           gaussian=self.gaussian)

        return model

def main():

    df_fluxes = pd.read_hdf('data/sdss/preprocessed/balanced_spectral_lines.h5', key='fluxes').head(3200)
    df_source_info = pd.read_hdf('data/sdss/preprocessed/balanced_spectral_lines.h5', key='source_info').head(3200)
    df_wavelengths = pd.read_hdf('data/sdss/preprocessed/balanced_spectral_lines.h5', key='wavelengths')

    mixed_input_model = MixedInputModel(gaussian=False,
                                        epochs=2,
                                        load_model=False,
                                        model_path='logs/mixed_input_gauss4_epoch32k.03-0.06.h5',
                                        spectral_lines=True)

    mixed_input_model.train(df_source_info, df_fluxes, df_wavelengths)

if __name__ == "__main__":
    main()