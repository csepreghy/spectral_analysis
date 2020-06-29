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

from sklearn.preprocessing import StandardScaler

from spectral_analysis.data_preprocessing.data_preprocessing import plot_spectrum, get_wavelengths_from_h5
from spectral_analysis.classifiers.neural_network.helper_functions import train_test_split
from spectral_analysis.plotify import Plotify

class AutoEncoder():
    def __init__(self, df_source_info, df_fluxes, df_wavelengths):
        X = self._prepare_data(df_source_info, df_fluxes, df_wavelengths)
        X_train, X_test = train_test_split(X, 0.2)
        
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        self.X_train = np.expand_dims(X_train, axis=2)
        self.X_test = np.expand_dims(X_test, axis=2)
        
        print(f'self.X_train = {self.X_train}')
        
        self.optimizer = Nadam(lr=0.001)

    
    def _prepare_data(self, df_source_info, df_fluxes, df_wavelengths):    
        df_quasars = df_source_info.loc[df_source_info['class'] == 'QSO']
        quasar_objids = df_quasars['objid'].to_numpy()
        quasar_fluxes = df_fluxes.loc[df_fluxes['objid'].isin(quasar_objids)]
        
        X = np.delete(quasar_fluxes.values, 0, axis=1)
        X = X[:, 0::8]
        print(f'X.shape = {X.shape}')

        X = X[:, np.mod(np.arange(X[0].size),25)!=0]

        print(f'{X.shape}')

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
        x = Conv1D(filters=16,
                   kernel_size=3,
                   activation='relu',
                   padding='same')(x)

        encoded = MaxPooling1D(2, padding="same")(x)

        # ================================================================================== #
        # ==================================== DECODER ===================================== #
        # ================================================================================== #

        x = Conv1D(filters=16,
                   kernel_size=3,
                   activation='relu',
                   padding='same')(encoded)
        
        x = UpSampling1D(2)(x)

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

        x = UpSampling1D(2)(x)
        decoded = Conv1D(1, 1, activation='relu', padding='same')(x)
        
        self.autoencoder = Model(input_layer, decoded)
        self.autoencoder.summary()
        self.autoencoder.compile(loss='mse', optimizer='adam')

        return self.autoencoder
    
    def train_model(self, epochs, batch_size=32):
        model = self.build_model()

        modelcheckpoint = ModelCheckpoint(filepath='logs/autoencoder.epoch{epoch:02d}.h5',
                                          monitor='val_loss',
                                          save_best_only=True)
        
        history = model.fit(x=self.X_train,
                            y=self.X_train,
                            epochs=epochs,
                            batch_size=32,
                            validation_data=(self.X_test, self.X_test),
                            callbacks=[EarlyStopping('val_loss', patience=8), modelcheckpoint])
        
        _, train_acc = model.evaluate(X_train, X_train, verbose=0)
        _, test_acc = model.evaluate(X_test, X_test, verbose=0)
        print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

        return model



    def evaluate_model(self, model):
        preds = model.predict(self.X_test)

        plotify = Plotify()
        _, axs = plotify.get_figax(nrows=2, figsize=(8, 10))
        axs[0].plot(self.wavelengths, self.X_test[24], color=plotify.c_orange)
        axs[1].plot(self.wavelengths, preds[24], color=plotify.c_orange)
        plt.savefig('plots/autoencoder_gaussian', dpi=160)
        plt.show()

        return preds

def main():
    df_fluxes = pd.read_hdf('data/sdss/preprocessed/balanced.h5', key='fluxes')
    df_source_info = pd.read_hdf('data/sdss/preprocessed/balanced.h5', key='source_info')
    df_wavelengths = pd.read_hdf('data/sdss/preprocessed/balanced.h5', key='wavelengths')

    ae = AutoEncoder(df_source_info, df_fluxes, df_wavelengths)
    model = ae.train_model(epochs=24, batch_size=64)

    ae.evaluate_model(model)
    

if __name__ == "__main__":
    main()