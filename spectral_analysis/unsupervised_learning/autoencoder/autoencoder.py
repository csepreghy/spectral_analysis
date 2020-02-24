import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time

from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, Reshape
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
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
        
        self.optimizer = Adam(lr=0.001)
        # self.build_model(hp=None)

    
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
        
        # ================================================================================== #
        # ==================================== ENCODER ===================================== #
        # ================================================================================== #
        
        input_layer = Input(shape=(self.X_train.shape[1], 1))

        # encoder
        x = Conv1D(filters=hp.Choice('encoder_conv_input_layer_filters', values=[8, 16, 32, 64, 128, 256]), 
                   kernel_size=hp.Choice('encoder_conv_input_layer_kernel_size', values=[2, 3, 4, 8, 16, 32, 64]),
                   activation='relu', 
                   padding='same')(input_layer)

        x = MaxPooling1D(2)(x)
        x = Conv1D(filters=hp.Choice(f'encoder_conv_1_layer_filters', values=[8, 16, 32, 64, 128]),
                    kernel_size=hp.Choice(f'encoder_conv_1_layer_kernel_size', values=[2, 3, 4, 8, 16, 32]),
                    activation='relu',
                    padding='same')(x)
        
        x = MaxPooling1D(2)(x)
        x = Conv1D(filters=hp.Choice(f'encoder_conv_2_layer_filters', values=[4, 8, 16, 32]),
                    kernel_size=hp.Choice(f'encoder_conv_2_layer_kernel_size', values=[2, 3, 4, 8, 16, 32]),
                    activation='relu',
                    padding='same')(x)

        x = MaxPooling1D(2)(x)
        x = Conv1D(filters=hp.Choice(f'encoder_conv_3_layer_filters', values=[4, 8, 12, 16]),
                    kernel_size=hp.Choice(f'encoder_conv_3_layer_kernel_size', values=[2, 3, 4, 8, 16]),
                    activation='relu',
                    padding='same')(x)

        x = MaxPooling1D(2)(x)
        x = Conv1D(filters=hp.Choice(f'encoder_conv_4_layer_filters', values=[2, 4, 8]),
                    kernel_size=hp.Choice(f'encoder_conv_4_layer_kernel_size', values=[2, 3, 4, 8]),
                    activation='relu',
                    padding='same')(x)

        encoded = MaxPooling1D(2, padding="same")(x)

        # ================================================================================== #
        # ==================================== DECODER ===================================== #
        # ================================================================================== #

        x = Conv1D(filters=hp.Choice('decoder_conv_input_layer_filters', values=[2, 4, 8]),
                   kernel_size=hp.Choice('decoder_conv_input_layer_kernel_size', values=[2, 3, 4, 8]),
                   activation='relu',
                   padding='same')(encoded)
        
        x = UpSampling1D(2)(x)

        x = Conv1D(filters=hp.Choice('decoder_conv_1_layer_filters', values=[4, 8, 12, 16]),
                   kernel_size=hp.Choice('decoder_conv_1_layer_kernel_size', values=[2, 3, 4, 8, 16]),
                   activation='relu',
                   padding='same')(x)

        x = UpSampling1D(2)(x)

        x = Conv1D(filters=hp.Choice('decoder_conv_2_layer_filters', values=[4, 8, 16, 32]),
                   kernel_size=hp.Choice('decoder_conv_2_layer_kernel_size', values=[2, 3, 4, 8, 16, 32]),
                   activation='relu',
                   padding='same')(x)

        x = UpSampling1D(2)(x)

        x = Conv1D(filters=hp.Choice('decoder_conv_3_layer_filters', values=[8, 16, 32, 64, 128]),
                   kernel_size=hp.Choice('decoder_conv_3_layer_kernel_size', values=[2, 3, 4, 8, 16, 32]),
                   activation='relu',
                   padding='same')(x)

        x = UpSampling1D(2)(x)

        x = Conv1D(filters=hp.Choice('decoder_conv_4_layer_filters', values=[8, 16, 32, 64, 128, 256]),
                   kernel_size=hp.Choice('decoder_conv_4_layer_kernel_size', values=[2, 3, 4, 8, 16, 32, 64]),
                   activation='relu',
                   padding='same')(x)

        x = UpSampling1D(2)(x)
        decoded = Conv1D(1, 1, activation='sigmoid', padding='same')(x) # 10 dims
        
        self.autoencoder = Model(input_layer, decoded)
        self.autoencoder.summary()
        self.autoencoder.compile(loss='mse', optimizer=self.optimizer)

        return self.autoencoder
    
    def train_model(self, epochs, batch_size=32):
        tuner = RandomSearch(self.build_model,
                             objective='val_loss',
                             max_trials=40,
                             executions_per_trial=1,
                             directory='logs/keras-tuner/',
                             project_name='autoencoder')
    
        tuner.search(x=self.X_train,
                     y=self.X_train,
                     epochs=24,
                     batch_size=32,
                     validation_data=(self.X_test, self.X_test))

        best_model = tuner.get_best_models(1)[0]
        best_hyperparameters = tuner.get_best_hyperparameters(1)[0]

        print(f'best_model = {best_model}')
        print(f'best_hyperparameters = {best_hyperparameters}')

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
    
    def eval_model(self):
        preds = self.autoencoder.predict(self.X_test)
        print(f'preds = {preds}')


        plotify = Plotify()
        fig, axs = plotify.get_figax(n_subplots=2)
        axs[0].plot(self.X_test[8], self.wavelengths)
        axs[1].plot(preds[8], self.wavelengths)
        plt.show()

        return preds

def main():
    df_fluxes = pd.read_hdf('data/sdss/preprocessed/0-50_gaussian.h5', key='fluxes')
    df_source_info = pd.read_hdf('data/sdss/preprocessed/0-50_gaussian.h5', key='spectral_data')

    ae = AutoEncoder(df_source_info, df_fluxes)
    ae.train_model(epochs=24, batch_size=64)

    ae.eval_model()
    

if __name__ == "__main__":
    main()