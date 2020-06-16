import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import History, EarlyStopping, TensorBoard
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical

from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

import time

LOG_DIR = f"{int(time.time())}"

class CNN:
	def __init__(self, df_fluxes, epochs):
		self.input_length = len(df_fluxes.columns) - 1
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
		model = self._build_model()
		print(model.summary())

		history = model.fit(x=X_train,
							y=y_train,
							epochs=self.epochs,
							batch_size=32,
							validation_data=(X_val, y_val),
							callbacks=[EarlyStopping('val_accuracy', patience=50),
									   TensorBoard(log_dir='logs/cnn')])

		# Evaluate Best Model #
		_, train_acc = model.evaluate(X_train, y_train, verbose=0)
		_, test_acc = model.evaluate(X_test, y_test, verbose=0)
		print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

	def _build_model(self):
		model = Sequential()

		model.add(Conv1D(filters=512,
						 kernel_size=7,
						 activation='relu',
						 input_shape=(self.input_length, 1)))
		
		model.add(Dropout(0.1))
		model.add(Conv1D(filters=512, kernel_size=5, activation='relu'))
		model.add(Dropout(0.1))
		model.add(MaxPooling1D(pool_size=2))
		model.add(Conv1D(filters=512, kernel_size=3, activation='relu'))
		model.add(Dropout(0.1))
		model.add(MaxPooling1D(pool_size=2))
		model.add(Conv1D(filters=256, kernel_size=7, activation='relu'))
		model.add(Dropout(0.1))
		model.add(MaxPooling1D(pool_size=2))
		model.add(Conv1D(filters=64, kernel_size=7, activation='relu'))
		model.add(Dropout(0.1))
		model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
		model.add(Dropout(0.1))
		model.add(Conv1D(filters=128, kernel_size=7, activation='relu'))
		model.add(Dropout(0.1))
		model.add(Conv1D(filters=512, kernel_size=3, activation='relu'))
		model.add(Dropout(0.1))

		model.add(Flatten())

		model.add(Dense(64, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(128, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(512, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(512, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(64, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(64, activation='relu'))
		model.add(Dropout(0.5))

		model.add(Dense(3, activation='softmax'))
		model.compile(loss='categorical_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])

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

    cnn = CNN(df_fluxes, epochs=2)
    cnn.run(df_source_info, df_fluxes)

if __name__ == "__main__":
    main()
