from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D
from keras.models import Model

window_length = 518

input_ts = Input(shape=(window_length,1))

x = Conv1D(32, 3, activation="relu", padding="valid")(input_ts)
x = MaxPooling1D(2, padding="valid")(x)

x = Conv1D(1, 3, activation="relu", padding="valid")(x)

encoded = MaxPooling1D(2, padding="valid")(x)

encoder = Model(input_ts, encoded)

x = Conv1D(16, 3, activation="relu", padding="valid")(encoded)
x = UpSampling1D(2)(x) 

x = Conv1D(32, 3, activation='relu', padding="valid")(x)
x = UpSampling1D(2)(x)

decoded = Conv1D(1, 1, activation='tanh', padding='valid')(x)

convolutional_autoencoder = Model(input_ts, decoded)

convolutional_autoencoder.summary()

optimizer = "nadam"
loss = "mean_absolute_error"

convolutional_autoencoder.compile(optimizer=optimizer, loss=loss)