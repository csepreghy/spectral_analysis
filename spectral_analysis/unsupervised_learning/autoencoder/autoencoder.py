import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, Reshape
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from spectral_analysis.data_preprocessing.data_preprocessing import remove_bytes_from_class, plot_spectrum, get_wavelengths_from_h5

class AutoEncoder():
    def __init__(self, df_source_info, df_fluxes, code_dimension):
        self.code_dimension = code_dimension

        self.X = self._prepare_data(df_source_info, df_fluxes)
        print(f'self.X.shape = {self.X.shape}')

        print(f'self.X = {type(self.X[0][0])}')
        self.X = np.expand_dims(self.X, axis=2)
        print(f'self.X.shape = {self.X.shape}')

        # print(f'self.X = {self.X}')

        optimizer = Adam(lr=0.001)

        self.autoencoder_model = self.build_model()
        self.autoencoder_model.compile(loss='mse', optimizer=optimizer)
    
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
        wavelengths = wavelengths[0:448]
        # plot_spectrum(X[0], wavelengths)
        return X
    
    def build_model(self):
        input_layer = Input(shape=(self.X.shape[1], 1))

        # encoder
        x= Conv1D(64, 4, activation='relu', padding='same')(input_layer)
        x = MaxPooling1D(2, padding="same")(x) # 5 dims
        x = Conv1D(32, 2, activation="relu", padding="same")(x) # 5 dims
        x = MaxPooling1D(2, padding='same')(x)
        x = Conv1D(16, 2, activation="relu", padding="same")(x) # 5 dims

        encoded = MaxPooling1D(2, padding="same")(x) # 3 dims
        encoder = Model(input_layer, encoded)

        # decoder
        x = Conv1D(16, 2, activation="relu", padding="same")(encoded) # 3 dims
        x = UpSampling1D(2)(x) # 6 dims
        x = Conv1D(32, 2, activation="relu", padding="same")(x) # 3 dims
        x = UpSampling1D(2)(x) # 6 dims
        x = Conv1D(64, 2, activation='relu', name='decoder_conv_2', padding='same')(x) # 5 dims
        x = UpSampling1D(2)(x) # 10 dims
        decoded = Conv1D(1, 1, activation='sigmoid', padding='same')(x) # 10 dims
        autoencoder = Model(input_layer, decoded)
        autoencoder.summary()

        return autoencoder
    
    def train_model(self, epochs, batch_size=20):
        early_stopping = EarlyStopping(monitor='val_loss',
                                       min_delta=0,
                                       patience=50,
                                       verbose=1, 
                                       mode='auto')

        history = self.autoencoder_model.fit(self.X, self.X,
                                             batch_size=batch_size,
                                             epochs=epochs,
                                             validation_data=(self.X, self.X),
                                             callbacks=[early_stopping])
        
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
    
    def eval_model(self, x_test):
        preds = self.autoencoder_model.predict(x_test)
        return preds

def main():
    df_fluxes = pd.read_hdf('data/sdss/preprocessed/0-50_gaussian.h5', key='fluxes')
    df_source_info = pd.read_hdf('data/sdss/preprocessed/0-50_gaussian.h5', key='spectral_data')

    # autoencoder = AutoEncoder(df_source_info=df_source_info, df_fluxes=df_fluxes, code_dimension=10)
    # autoencoder.encoder_decoder()
    # autoencoder.fit(batch_size=32, epochs=100)

    ae = AutoEncoder(df_source_info, df_fluxes, 10)
    ae.train_model(epochs=200, batch_size=20)


if __name__ == "__main__":
    main()