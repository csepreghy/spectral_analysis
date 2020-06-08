import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import time as time
import datetime
import math
import seaborn as sn
import random

# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import History, TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Activation, Input, concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical

from spectral_analysis.spectral_analysis.plotify import Plotify
from spectral_analysis.spectral_analysis.data_preprocessing.bpt_diagram import plot_bpt_diagram




# This is a mixed input neural network that combines a CNN with an MLP.
# Inputs:
#    - df: pandas dataframe with training data
#    - batch_size: batch size (integer)
#    - hidden_layers: an array of numbers that represent the number of hidden layers and the
#                     number of neurons in each. [128, 128, 128] is 3 hidden layers with 128
#                     neurons each
#    - n_ephoch: the number of epochs the system trains
#
# It then prepares, trains and saves the model to disk so you can load it later. Currently it is 
# a binary classifier, but it can be easily changed
# It also automatically scales the data. This should speed up the process of training

def train_test_split(X, test_size, y=None, objids=None, indeces=None):
    if y is not None and len(X) != len(y): assert('X and y does not have the same length')

    n_test = round(len(X) * test_size)
    n_train = len(X) - n_test

    X_test = X[-n_test:]
    X_train = X[:n_train]

    if indeces is not None:
        i_test = indeces[-n_test:]
        i_train = indeces[:n_train]

    # print('len(X_train)', len(X_train))
    # print('len(i_train)', len(i_train))
    # print('len(X_test)', len(X_test))
    # print('len(i_test)', len(i_test))

    if y is not None:
        y_test = y[-n_test:]
        y_train = y[:n_train]

    if y is not None: return X_train, X_test, y_train, y_test, i_train, i_test

    else: return X_train, X_test

def get_incorrect_predictions(model, X_test_fluxes, X_test_spectra, raw_X_test_spectra, y_test, df_source_info_test, df_wavelengths, gaussian=False):
    classes = ['GALAXY', 'QSO', 'STAR']
    predictions = model.predict(X_test_fluxes).argmax(axis=1)
    print(f'predictions = {predictions[0:200]}')
    y_test = y_test.argmax(axis=1)
    wrong_indeces = []
    correct_indeces = []
    for i in range(len(predictions)):
        if predictions[i] != y_test[i]:
            wrong_indeces.append(i)
        
        else: correct_indeces.append(i)

    # indices = [i for i in enumerate(predictions) if predictions[i] != y_test[i]]
    wrong_predictions = []
    for i in wrong_indeces:
        wrong_prediction = {'spectrum': X_test_spectra[i],
                            'raw_spectrum': raw_X_test_spectra[i],
                            'predicted': classes[predictions[i]],
                            'target_class': classes[y_test[predictions[i]]]}

        wrong_predictions.append(wrong_prediction)
    
    correct_predictions = []
    for i in correct_indeces:
        correct_prediction = {'spectrum': X_test_spectra[i],
                              'raw_spectrum': raw_X_test_spectra[i],
                              'predicted': classes[predictions[i]],
                              'target_class': classes[y_test[predictions[i]]]}

        correct_predictions.append(correct_prediction)

    plotify = Plotify(theme='ugly')

    for i, wrong_prediction in enumerate(wrong_predictions[0:50]):
        fluxes = wrong_predictions[i]['spectrum']
        raw_fluxes = wrong_predictions[i]['raw_spectrum']
        wavelengths = df_wavelengths.values
        source_info = df_source_info_test.iloc[i]

        if gaussian == False:
            fig, ax = plotify.get_figax()
            title = f'ra = {source_info.get(["ra"][0])}, dec = {source_info.get(["dec"][0])}, plate = {source_info.get(["plate"][0])}\n\n Predicted: {wrong_prediction["predicted"]}, Target Class: {wrong_prediction["target_class"]}'
            ax.set_title(title, pad=10, fontsize=13)
            ax.set_xlabel('Wavelength (Å)')
            ax.set_ylabel(r'$F_{\lambda[10^{-17} erg \: cm^{-2}s^{-1} Å^{-1}]}$', fontsize=14)
            plt.plot(wavelengths, fluxes, color=plotify.c_orange, lw=0.6)
            plt.savefig(f'plots/wrong_predictions/wrong_prediction_{i}.png', dpi=150)
        
        if gaussian == True:
            _, axs = plotify.get_figax(nrows=2, figsize=(8, 8))
            axs[0].plot(wavelengths, raw_fluxes, color=plotify.c_orange, lw=0.6)
            axs[1].plot(wavelengths, fluxes, color=plotify.c_orange, lw=0.6)
            
            title = f'ra = {source_info.get(["ra"][0])}, dec = {source_info.get(["dec"][0])}, plate = {source_info.get(["plate"][0])}\n\n Predicted: {wrong_prediction["predicted"]}, Target Class: {wrong_prediction["target_class"]}'

            axs[0].set_title(title, pad=10, fontsize=13)
            axs[1].set_title(r'with Gaussian convolution, $\sigma = 4$')
            axs[0].set_ylabel(r'$F_{\lambda[10^{-17} erg \: cm^{-2}s^{-1} Å^{-1}]}$', fontsize=14)
            axs[1].set_ylabel(r'$F_{\lambda[10^{-17} erg \: cm^{-2}s^{-1} Å^{-1}]}$', fontsize=14)
            axs[1].set_xlabel('Wavelength (Å)')

            plt.subplots_adjust(hspace=0.4)
            plt.savefig(f'plots/wrong_predictions/gaussian8_wrong_prediction_{i}.png', dpi=150)

    
    for i, correct_prediction in enumerate(correct_predictions[0:50]):
        fluxes = correct_predictions[i]['spectrum']
        raw_fluxes = correct_predictions[i]['raw_spectrum']
        wavelengths = df_wavelengths.values
        fig, ax = plotify.get_figax()
        source_info = df_source_info_test.iloc[i]

        if gaussian == False:
            fig, ax = plotify.get_figax()
            title = f'ra = {source_info.get(["ra"][0])}, dec = {source_info.get(["dec"][0])}, plate = {source_info.get(["plate"][0])}\n\n Predicted: {correct_prediction["predicted"]}, Target Class: {correct_prediction["target_class"]}'
            ax.set_title(title, pad=10, fontsize=13)
            ax.set_xlabel('Wavelength (Å)')
            ax.set_ylabel(r'$F_{\lambda[10^{-17} erg \: cm^{-2}s^{-1} Å^{-1}]}$', fontsize=14)
            plt.plot(wavelengths, fluxes, color=plotify.c_orange, lw=0.6)
            plt.savefig(f'plots/correct_predictions/correct_prediction_{i}.png', dpi=150)
        
        if gaussian == True:
            fig, axs = plotify.get_figax(nrows=2, figsize=(8, 8))
            axs[0].plot(wavelengths, raw_fluxes, color=plotify.c_orange, lw=0.6)
            axs[1].plot(wavelengths, fluxes, color=plotify.c_orange, lw=0.6)
            
            title = f'ra = {source_info.get(["ra"][0])}, dec = {source_info.get(["dec"][0])}, plate = {source_info.get(["plate"][0])}\n\n Predicted: {correct_prediction["predicted"]}, Target Class: {correct_prediction["target_class"]}'

            axs[0].set_title(title, pad=10, fontsize=13)
            axs[1].set_title(r'with Gaussian convolution, $\sigma = 4$')
            axs[0].set_ylabel(r'$F_{\lambda[10^{-17} erg \: cm^{-2}s^{-1} Å^{-1}]}$', fontsize=14)
            axs[1].set_ylabel(r'$F_{\lambda[10^{-17} erg \: cm^{-2}s^{-1} Å^{-1}]}$', fontsize=14)
            axs[1].set_xlabel('Wavelength (Å)')

            plt.subplots_adjust(hspace=0.4)
            plt.savefig(f'plots/correct_predictions/gaussian8_correct_prediction_{i}.png', dpi=150)

	
    # plt.show()

def evaluate_model(model, X_test, y_test, df_source_info, indeces, classes):
    labels = []
    for label in classes:
        if label == 'label_': labels.append('NULL')
        else: labels.append(label.replace('label_', ''))

    print(f'labels = {labels}')
    y_pred = model.predict(X_test)

    print(f'X_test = {len(X_test)}')
    print(f'indeces = {len(indeces)}')
    print(f'len(y_pred) = {len(y_pred)}')

    matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    print(f'confusion matrix: \n {matrix}')

    print(f'df_source_info[indeces] = {len(df_source_info.loc[indeces])}')

    # plot_bpt_diagram(df_source_info.loc[indeces], labels=labels, y_pred=y_pred, y_test=y_test)

    # df_cm = pd.DataFrame(matrix,
    # 					 index=[i for i in classes],
    # 					 columns=[i for i in classes])

    # fig, ax = plt.subplots(figsize=(10,7))
    # sn.heatmap(df_cm, annot=True, annot_kws={"size": 14})
    # ax.set_ylabel('Predicted Class', color='black')
    # ax.set_xlabel('Target Class', color='black')
    # ax.set_title('Confusion Matrix')
    # plt.show()
  
def shuffle_in_unison(a, b, indeces):
    print(f'a.shape = {a.shape}')
    print(f'b.shape = {b.shape}')
    arr = np.array([10, 20, 30, 40, 50])
    idx = [1, 0, 3, 4, 2]
    arr[idx]

    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    np.random.set_state(rng_state)
    np.random.shuffle(indeces)

    return a, b, indeces

def shuffle_along_axis(a, axis):
    a = np.array(a)
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)

def main():
    a = np.array([['A1', 'A2', 'A3', 'A4'],
         ['B1', 'B2', 'B3', 'B4'],
         ['C1', 'C2', 'C3', 'C4'],
         ['D1', 'D2', 'D3', 'D4']])
    
    b = np.array([['AA1', 'AA2', 'AA3', 'AA4'],
         ['BB1', 'BB2', 'BB3', 'BB4'],
         ['CC1', 'CC2', 'CC3', 'CC4'],
         ['DD1', 'DD2', 'DD3', 'DD4']])
    
    c = [1, 2, 3, 4]

    a, b, c = shuffle_in_unison(a, b, c)
    print(a)
    print(b)
    print(c)

    # train = pd.read_csv('data/tensorboard-logs/mixed-input_train-tag-epoch_accuracy.csv')['Value'].values
    # validation = pd.read_csv('data/tensorboard-logs/mixed-input_validation-tag-epoch_accuracy.csv')['Value'].values
    # xs = list(range(33))

    # plotify = Plotify(theme='ugly')
    # fig, ax = plotify.get_figax()


    # ax.plot(xs, train)
    # plt.show()

if __name__ == "__main__":
	main()