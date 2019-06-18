import pandas as pd
import time as time

from src.import_data import get_save_SDSS_from_coordinates
from src.SDSS_direct_query import query
from src.merge_tables import merge

from src.data_preprocessing import filter_sources, spectrum_cutoff, create_continuum, merge_lines_and_continuum
from src.get_spectrallines import get_spectrallines
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from matplotlib import style

style.use('fivethirtyeight')

from src.plotify import Plotify

plotify = Plotify()
# with open('data/sdss_coordinates_lowz.txt') as text_file:
#   coord_list = text_file.read().splitlines()
#     mystring.replace('\n', ' ').replace('\r', '')

# query()

# coord_list=pd.read_csv("data/lowz.csv")
# start=time.time()
# ra_list = coord_list["ra"].tolist()
# dec_list= coord_list["dec"].tolist()
# end=time.time()
# tt=end - start
# print("time for listing is:", tt)

# start1=time.time()
# ra=ra_list[40001:45000]
# dec=dec_list[40001:45000]
# get_save_SDSS_from_coordinates( ra , dec )
# end1=time.time()

# tt1= end1- start1
# length=len(ra) -1
# print("time for "+str(length)+" stellar objects:" , tt1)

# merge(length)

# start = time.time()

# spectra = pd.read_pickle('data/sdss/FinalTable_Andrew(40-45).pkl')
# df_filtered = filter_sources(df = spectra)
# print('DF Filtered: ')
# print(df_filtered.head())
# df_spectral_lines = get_spectrallines(df_filtered)
# print('Spectral Lines')
# print(df_spectral_lines.head())
# df_spectral_lines.to_pickle('spectral_lines_df.pkl')
# df_cutoff = spectrum_cutoff(df = df_filtered)
# df_continuum = create_continuum(df = df_cutoff, sigma=8, downsize=8)
# df_continuum.to_pickle('continuum_df.pkl')

# df_complete = merge_lines_and_continuum(df_spectral_lines, df_continuum)
# df_complete.to_pickle('COMPLETE_df.pkl')

# # end = time.time()
# # tt = end - start
# # print(" ")
# # print("Time elapsed: ", tt, "s")
# # print(tt/60, "min")

# # model = create_model(df_final, configs['neural_network'])


# generate 2d classification dataset
scaler = StandardScaler()

# train, test = train_test_split(df, test_size=0.2)
# train, val = train_test_split(train, test_size=0.2)

columns = []

df = pd.read_pickle('COMPLETE_df.pkl')

df['class'] = pd.Categorical(df['class'])
dfDummies = pd.get_dummies(df['class'], prefix='category')
df = pd.concat([df, dfDummies], axis=1)

for column in df.columns:
  if column not in ['class', 'dec', 'ra', 'plate', 'wavelength', 'objid']:
    columns.append(column)

print('columns', columns)

X = []
y = []

for index, spectrum in df[columns].iterrows():
  X_row = []

  spectral_lines = spectrum['spectral_lines']
  for spectral_line in spectral_lines: X_row.append(spectral_line)

  flux_list = spectrum['flux_list']
  for flux in flux_list:
    X_row.append(flux)
  
  X_row.append(spectrum['z'])
  X_row.append(spectrum['zErr'])

  category_GALAXY = spectrum['category_GALAXY']
  category_QSO = spectrum['category_QSO']
  category_STAR = spectrum['category_STAR']

  y_row = [category_GALAXY, category_QSO, category_STAR]

  X.append(X_row)
  y.append(y_row)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print('X_train', X_train[0])
print('X_train_std', X_train_std)

X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)
X_val_std = scaler.transform(X_val)

# define model
model = Sequential()
model.add(Dense(256, input_dim=X_train_std.shape[1], activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(256, input_dim=256, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(256, input_dim=256, activation='relu', kernel_initializer='he_uniform'))

model.add(Dense(3, activation='softmax'))
opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
# fit model


#spectra = pd.read_pickle('data/alldatamerged.pkl')

#df_continuum = pd.read_pickle('continuum_df.pkl')
df_spectral_lines = pd.read_pickle('spectral_lines_df.pkl')


start = time.time()


spectra = pd.read_pickle('data/alldatamerged.pkl')
df_filtered = filter_sources(df = spectra)
print('DF Filtered: ')
print(df_filtered.head())
"""
df_spectral_lines = get_spectrallines(df_filtered)
print('Spectral Lines')
print(df_spectral_lines.head())
df_spectral_lines.to_pickle('spectral_lines2_df.pkl')
"""
df_cutoff = spectrum_cutoff(df = df_filtered)
df_continuum = create_continuum(df = df_cutoff, sigma=16, downsize=8)
df_continuum.to_pickle('continuum_df.pkl')


df_complete = merge_lines_and_continuum(df_spectral_lines, df_continuum)
print("DF Complete: ")
print(df_complete.head())
df_complete.to_pickle('COMPLETE_df.pkl')

y_train = np.array(y_train)
y_test = np.array(y_test)
print('y_train.shape', y_train.shape)
print('type(y_train)', type(y_train))

history = model.fit(X_train_std, y_train, validation_data=(X_test_std, y_test), epochs=100, verbose=0)

# evaluate the model
_, train_acc = model.evaluate(X_train_std, y_train, verbose=0)
_, test_acc = model.evaluate(X_test_std, y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
# plot loss during training
plt.subplot(211)
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
# plot accuracy during training
plt.subplot(212)
plt.title('Accuracy')

plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()
