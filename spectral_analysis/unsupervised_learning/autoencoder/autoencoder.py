import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time

from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, Reshape
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping

from kerastuner.engine.hyperparameters import HyperParameters
from kerastuner.tuners import RandomSearch

from sklearn.preprocessing import StandardScaler

from spectral_analysis.data_preprocessing.data_preprocessing import remove_bytes_from_class, plot_spectrum, get_wavelengths_from_h5
from spectral_analysis.classifiers.neural_network.helper_functions import train_test_split
from spectral_analysis.plotify import Plotify

class AutoEncoder():
    def __init__(self, df_source_info, df_fluxes):
        X = self._prepare_data(df_source_info, df_fluxes)
        X_train, X_test = train_test_split(X, 0.2)
        
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        self.X_train = np.expand_dims(X_train, axis=2)
        self.X_test = np.expand_dims(X_test, axis=2)
        
        print(f'self.X_train = {self.X_train}')
        
        self.optimizer = Nadam(lr=0.001)

    
    def _prepare_data(self, df_source_info, df_fluxes):
        if "b'" in df_source_info['class'][0]:
            df_source_info = remove_bytes_from_class(df_source_info)
    
        df_quasars = df_source_info.loc[df_source_info['class'] == 'QSO']
        quasar_objids = df_quasars['objid'].to_numpy()
        quasar_fluxes = df_fluxes.loc[df_fluxes['objid'].isin(quasar_objids)]
        
        X = np.delete(quasar_fluxes.values, 0, axis=1)
        X = X[:, 0::8]
        print(f'X.shape = {X.shape}')

        X = X[:, np.mod(np.arange(X[0].size),25)!=0]

        print(f'{X.shape}')

        wavelengths = get_wavelengths_from_h5(filename='/sdss/preprocessed/0-50k_original_fluxes.h5')
        wavelengths = wavelengths[::8]
        self.wavelengths = wavelengths[0:448]
        # plot_spectrum(X[0], wavelengths)
        return X
    
    def build_model(self, hp):

        hyperparameters = {
            'layer_1_filters': hp.Choice('layer_1_filters', values=[8, 16, 32, 64, 128, 256]),
            'layer_1_kernel_size': hp.Choice('layer_1_kernel_size', values=[2, 3, 4, 8, 16, 32, 64]),
            'layer_2_filters': hp.Choice('layer_2_filters', values=[8, 16, 32, 64, 128]),
            'layer_2_kernel_size': hp.Choice('layer_2_kernel_size', values=[2, 3, 4, 8, 16, 32]),
            'layer_3_filters': hp.Choice('layer_3_filters', values=[4, 8, 16, 32]),
            'layer_3_kernel_size': hp.Choice('layer_3_kernel_size', values=[2, 3, 4, 8, 16, 32]),
            'layer_4_filters': hp.Choice('layer_4_filters', values=[4, 8, 12, 16]),
            'layer_4_kernel_size': hp.Choice('layer_4_kernel_size', values=[2, 3, 4, 8]),
            'layer_5_filters': hp.Choice('layer_5_filters', values=[2, 3, 4]),
            'layer_5_kernel_size': hp.Choice('layer_5_kernel_size', values=[2, 3, 4])
        }
        
        # ================================================================================== #
        # ==================================== ENCODER ===================================== #
        # ================================================================================== #
        
        input_layer = Input(shape=(self.X_train.shape[1], 1))

        # encoder
        x = Conv1D(filters=hyperparameters['layer_1_filters'],
                   kernel_size=hyperparameters['layer_1_kernel_size'],
                   activation='relu', 
                   padding='same')(input_layer)

        x = MaxPooling1D(2)(x)
        x = Conv1D(filters=hyperparameters['layer_2_filters'],
                    kernel_size=hyperparameters['layer_2_kernel_size'],
                    activation='relu',
                    padding='same')(x)
        
        x = MaxPooling1D(2)(x)
        x = Conv1D(filters=hyperparameters['layer_3_filters'],
                    kernel_size=hyperparameters['layer_3_kernel_size'],
                    activation='relu',
                    padding='same')(x)

        x = MaxPooling1D(2)(x)
        x = Conv1D(filters=hyperparameters['layer_4_filters'],
                    kernel_size=hyperparameters['layer_4_kernel_size'],
                    activation='relu',
                    padding='same')(x)

        x = MaxPooling1D(2)(x)
        x = Conv1D(filters=hyperparameters['layer_5_filters'],
                    kernel_size=hyperparameters['layer_5_kernel_size'],
                    activation='relu',
                    padding='same')(x)

        encoded = MaxPooling1D(2, padding="same")(x)

        # ================================================================================== #
        # ==================================== DECODER ===================================== #
        # ================================================================================== #

        x = Conv1D(filters=hyperparameters['layer_5_filters'],
                   kernel_size=hyperparameters['layer_5_kernel_size'],
                   activation='relu',
                   padding='same')(encoded)
        
        x = UpSampling1D(2)(x)

        x = Conv1D(filters=hyperparameters['layer_4_filters'],
                   kernel_size=hyperparameters['layer_4_kernel_size'],
                   activation='relu',
                   padding='same')(x)

        x = UpSampling1D(2)(x)

        x = Conv1D(filters=hyperparameters['layer_3_filters'],
                   kernel_size=hyperparameters['layer_3_kernel_size'],
                   activation='relu',
                   padding='same')(x)

        x = UpSampling1D(2)(x)

        x = Conv1D(filters=hyperparameters['layer_2_filters'],
                   kernel_size=hyperparameters['layer_2_kernel_size'],
                   activation='relu',
                   padding='same')(x)

        x = UpSampling1D(2)(x)

        x = Conv1D(filters=hyperparameters['layer_1_filters'],
                   kernel_size=hyperparameters['layer_1_kernel_size'],
                   activation='relu',
                   padding='same')(x)

        x = UpSampling1D(2)(x)
        decoded = Conv1D(1, 1, activation='sigmoid', padding='same')(x)
        
        self.autoencoder = Model(input_layer, decoded)
        self.autoencoder.summary()
        self.autoencoder.compile(loss='mse', optimizer=self.optimizer)

        return self.autoencoder
    
    def train_model(self, epochs, batch_size=32):
        self.tuner = RandomSearch(self.build_model,
                                  objective='val_loss',
                                  max_trials=8,
                                  executions_per_trial=1,
                                  directory='logs/keras-tuner/',
                                  project_name='autoencoder')
    
        self.tuner.search(x=self.X_train,
                          y=self.X_train,
                          epochs=18,
                          batch_size=32,
                          validation_data=(self.X_test, self.X_test))

        # history = self.autoencoder.fit(self.X_train, self.X_train,
        #                                batch_size=batch_size,
        #                                epochs=epochs,
        #                                validation_data=(self.X_test, self.X_test))
  
        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.title('Model loss')
        # plt.ylabel('Loss')
        # plt.xlabel('Epoch')
        # plt.legend(['Train', 'Test'], loc='upper left')
        # plt.show()
    
    def evaluate_model(self):
        best_model = self.tuner.get_best_models(1)[0]
        best_hyperparameters = self.tuner.get_best_hyperparameters(1)[0]

        print(f'best_model = {best_model}')
        print(f'best_hyperparameters = {best_hyperparameters}')

        preds = best_model.predict(self.X_test)
        print(f'preds = {preds}')

        plotify = Plotify()
        fig, axs = plotify.get_figax(nrows=2)
        axs[0].plot(self.wavelengths, self.X_test[24])
        axs[1].plot(self.wavelengths, preds[24])
        plt.show()

        return preds

def main():
    df_fluxes = pd.read_hdf('data/sdss/preprocessed/0-50_gaussian.h5', key='fluxes')
    df_source_info = pd.read_hdf('data/sdss/preprocessed/0-50_gaussian.h5', key='spectral_data')

    ae = AutoEncoder(df_source_info, df_fluxes)
    ae.train_model(epochs=24, batch_size=64)

    ae.evaluate_model()
    

if __name__ == "__main__":
    main()