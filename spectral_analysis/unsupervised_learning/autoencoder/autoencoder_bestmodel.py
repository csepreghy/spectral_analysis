import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time

from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, Reshape
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import TensorBoard, History, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping

from kerastuner.engine.hyperparameters import HyperParameters
from kerastuner.tuners import RandomSearch

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from spectral_analysis.spectral_analysis.data_preprocessing.data_preprocessing import plot_spectrum, get_wavelengths_from_h5
from spectral_analysis.spectral_analysis.classifiers.neural_network.helper_functions import train_test_split
from spectral_analysis.spectral_analysis.plotify import Plotify

class AutoEncoder():
    def __init__(self, df_source_info, df_fluxes, df_wavelengths, load_model, weights_path=''):
        self.load_model = load_model
        self.weights_path = weights_path
        X = self._prepare_data(df_source_info, df_fluxes, df_wavelengths)
        X_train, X_test = train_test_split(X, 0.2)
        X_train, X_val = train_test_split(X_train, 0.2)
        self.objids_train, self.objids_test = train_test_split(self.quasar_objids, 0.2)
        
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        X_val = self.scaler.transform(X_val)

        self.X_train = np.expand_dims(X_train, axis=2)
        self.X_test = np.expand_dims(X_test, axis=2)
        self.X_val = np.expand_dims(X_val, axis=2)

    
    def _prepare_data(self, df_source_info, df_fluxes, df_wavelengths):    
        self.df_quasars = df_source_info.loc[df_source_info['class'] == 'QSO']
        self.quasar_objids = self.df_quasars['objid'].to_numpy()
        quasar_fluxes = df_fluxes.loc[df_fluxes['objid'].isin(self.quasar_objids)]
        
        X = np.delete(quasar_fluxes.values, 0, axis=1)
        X = X[:, 0::8]
        print(f'X.shape = {X.shape}')

        X = X[:, np.mod(np.arange(X[0].size),25)!=0]

        wavelengths = df_wavelengths.to_numpy()
        wavelengths = wavelengths[::8]
        self.wavelengths = wavelengths[0:448]
        # plot_spectrum(X[0], wavelengths)
        return X
    
    def build_model(self):
        # ================================================================================== #
        # ==================================== ENCODER ===================================== #
        # ================================================================================== #
        
        input_layer = Input(shape=(self.X_train.shape[1], 1))

        # encoder
        x = Conv1D(filters=256,
                   kernel_size=7,
                   activation='relu', 
                   padding='same')(input_layer)
        x = MaxPooling1D(2)(x)

        x = Conv1D(filters=128,
                   kernel_size=5,
                   activation='relu',
                   padding='same')(x)
        x = MaxPooling1D(2)(x)
        
        x = Conv1D(filters=64,
                   kernel_size=5,
                   activation='relu',
                   padding='same')(x)
        x = MaxPooling1D(2)(x)
        
        x = Conv1D(filters=32,
                   kernel_size=3,
                   activation='relu',
                   padding='same')(x)
        x = MaxPooling1D(2)(x)
        
        encoded = Conv1D(1, 1, activation='relu', padding="same")(x)

        # ================================================================================== #
        # ==================================== DECODER ===================================== #
        # ================================================================================== #
        
        x = UpSampling1D(2)(encoded)

        x = Conv1D(filters=32,
                   kernel_size=3,
                   activation='relu',
                   padding='same')(x)

        x = UpSampling1D(2)(x)

        x = Conv1D(filters=64,
                   kernel_size=5,
                   activation='relu',
                   padding='same')(x)

        x = UpSampling1D(2)(x)

        x = Conv1D(filters=128,
                   kernel_size=5,
                   activation='relu',
                   padding='same')(x)

        x = UpSampling1D(2)(x)

        x = Conv1D(filters=256,
                   kernel_size=7,
                   activation='relu',
                   padding='same')(x)

        decoded = Conv1D(1, 1, activation='tanh', padding='same')(x)
        
        self.autoencoder = Model(input_layer, decoded)
        self.autoencoder.summary()
        self.autoencoder.compile(loss='mse', optimizer='adam')

        return self.autoencoder
    
    def train_model(self, epochs, batch_size=32):
        model = self.build_model()
        
        if self.load_model == False:
            modelcheckpoint = ModelCheckpoint(filepath='logs/autoencoder.epoch{epoch:02d}.h5',
                                              monitor='val_loss',
                                              save_best_only=True)
            
            history = model.fit(x=self.X_train,
                                y=self.X_train,
                                epochs=epochs,
                                batch_size=32,
                                validation_data=(self.X_val, self.X_val),
                                callbacks=[EarlyStopping('val_loss', patience=8), modelcheckpoint])

            self.evaluate_model(model)

        else:
            model.load_weights(self.weights_path)
            print(f'model = {model}')
            bottleneck = model.get_layer('conv1d_5')

            extractor = Model(inputs=model.inputs, outputs=[bottleneck.output])
            features = extractor(self.X_test)

            print('extractor', extractor)
            print('features', features)
            print('features.shape', features.shape)
            # self.evaluate_model(model)

        return model

    def evaluate_model(self, model):
        preds = model.predict(self.X_test)
        
        print(self.X_test.shape)
        self.X_test = np.squeeze(self.X_test, axis=2)
        preds = np.squeeze(preds, axis=2)
        print(self.X_test.shape)

        self.X_test = self.scaler.inverse_transform(self.X_test)
        preds = self.scaler.inverse_transform(preds)
        
        for i in range(50):
            qso_ra = self.df_quasars.loc[self.df_quasars['objid'] == self.objids_test[i]]['ra'].values[0]
            qso_dec = self.df_quasars.loc[self.df_quasars['objid'] == self.objids_test[i]]['dec'].values[0]
            qso_plate = self.df_quasars.loc[self.df_quasars['objid'] == self.objids_test[i]]['plate'].values[0]
            qso_z = self.df_quasars.loc[self.df_quasars['objid'] == self.objids_test[i]]['z'].values[0]

            plotify = Plotify(theme='ugly')
            _, axs = plotify.get_figax(nrows=2, figsize=(6, 8))
            axs[0].plot(self.wavelengths, self.X_test[i], color=plotify.c_orange)
            axs[1].plot(self.wavelengths, preds[i], color=plotify.c_orange)
            axs[0].set_title(f'ra = {qso_ra}, dec = {qso_dec}, z = {qso_z}, plate = {qso_plate}', fontsize=14)
            axs[1].set_title(f'Autoencoder recreation')
            axs[0].set_ylabel(r'$F_{\lambda[10^{-17} erg \: cm^{-2}s^{-1} Å^{-1}]}$', fontsize=14)
            axs[1].set_ylabel(r'$F_{\lambda[10^{-17} erg \: cm^{-2}s^{-1} Å^{-1}]}$', fontsize=14)
            axs[1].set_xlabel('Wavelength (Å)')

            plt.subplots_adjust(hspace=0.4)
            plt.savefig(f'plots/autoencoder/autoencoder_gaussian{i}', dpi=160)

        return preds

def main():
    df_fluxes = pd.read_hdf('data/sdss/preprocessed/balanced.h5', key='fluxes')
    df_source_info = pd.read_hdf('data/sdss/preprocessed/balanced.h5', key='source_info')
    df_wavelengths = pd.read_hdf('data/sdss/preprocessed/balanced.h5', key='wavelengths')

    ae = AutoEncoder(df_source_info, df_fluxes, df_wavelengths, load_model=False, weights_path='logs/colab-logs/autoencoder.epoch65.h5')
    ae.train_model(epochs=12, batch_size=64)
    

if __name__ == "__main__":
    main()