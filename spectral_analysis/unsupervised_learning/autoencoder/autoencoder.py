import numpy as np
import pandas as pd
import os

from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard

from spectral_analysis.data_preprocessing.data_preprocessing import remove_bytes_from_class, plot_spectrum

class AutoEncoder():
    def __init__(self, df_source_info, df_fluxes, code_dimension):
        self.code_dimension = code_dimension
        self._prepare_data(df_source_info, df_fluxes)
    
    def _prepare_data(self, df_source_info, df_fluxes):
        if "b'" in df_source_info['class'][0]:
            df_source_info = remove_bytes_from_class(df_source_info)
    
        df_quasars = df_source_info.loc[df_source_info['class'] == 'QSO']
        quasar_objids = df_quasars['objid'].to_numpy()
        quasar_fluxes = df_fluxes.loc[df_fluxes['objid'].isin(quasar_objids)]
        r = lambda: np.random.randint(1, 3)
        self.X = np.delete(quasar_fluxes.values, 0, axis=1)
        # self.X = np.array([[r(), r(), r()] for _ in range(1000)])  # create data to pass in. The input.
        print(f'{self.X}')

        # plot_spectrum(self.X[0])


    def _encoder(self):
        inputs = Input(shape=(self.X[0].shape))
        # encoded = Dense(self.code_dimension, activation='relu')(inputs)

        x = Conv1D(filters=32, kernel_size=3, activation='relu')(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(filters=32, kernel_size=3, activation='relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(filters=32, kernel_size=3, activation='relu')(x)
        encoded = MaxPooling1D(pool_size=2)(x)

        model = Model(inputs, encoded)
        self.encoder = model

        return model
    
    def _decoder(self):
        inputs = Input(shape=(self.code_dimension,))
        # decoded = Dense(self.X[0].shape[0])(inputs)

        x = Conv1D(filters=32, kernel_size=3, activation='relu')(inputs)
        x = UpSampling1D(2)(x)
        x = Conv1D(filters=32, kernel_size=3, activation='relu')(x)
        x = UpSampling1D(2)(x)
        x = Conv1D(filters=32, kernel_size=3, activation='relu')(x)
        x = UpSampling1D(2)(x)
        
        decoded = Conv1D(filters=1, pool_size=2, activation='sigmoid')(x)

        model = Model(inputs, decoded)
        self.decoder = model

        return model

    def encoder_decoder(self):
        encoder_model = self._encoder()
        decoder_model = self._decoder()

        inputs = Input(shape=self.X[0].shape)
        encoder_output = encoder_model(inputs)
        decoder_output = decoder_model(encoder_output)

        self.model = Model(inputs, decoder_output)

    def fit(self, batch_size, epochs):
        self.model.compile(optimizer='sgd', loss='mse')
        self.model.fit(self.X, self.X, epochs=epochs, batch_size=batch_size)

def main():
    df_fluxes = pd.read_hdf('data/sdss/preprocessed/0-50_gaussian.h5', key='fluxes')
    df_source_info = pd.read_hdf('data/sdss/preprocessed/0-50_gaussian.h5', key='spectral_data')

    autoencoder = AutoEncoder(df_source_info=df_source_info, df_fluxes=df_fluxes, code_dimension=10)
    autoencoder.encoder_decoder()
    autoencoder.fit(batch_size=32, epochs=100)


if __name__ == "__main__":
    main()