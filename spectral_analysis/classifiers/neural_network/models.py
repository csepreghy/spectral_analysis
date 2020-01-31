from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import History
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical

def create_cnn(input_length):
  model = Sequential()

  model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(input_length, 1)))
  model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
  model.add(Dropout(0.5))
  model.add(MaxPooling1D(pool_size=2))
  model.add(Flatten())
  model.add(Dense(32, activation='relu'))

  return model

def create_mlp(input_shape):
  model = Sequential()

  model.add(Dense(256, input_dim=input_shape, activation='relu', kernel_initializer='he_uniform'))
  model.add(Dense(256, input_dim=256, activation='relu', kernel_initializer='he_uniform'))

  return model

