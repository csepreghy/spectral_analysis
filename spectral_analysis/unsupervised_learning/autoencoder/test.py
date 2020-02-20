from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
import numpy as np  # Scientific computing libarary (we will use for converting our data to NumPy arrays)
import os


class AutoEncoder:  # define the overall template for the model.
    def __init__(self, encoding_dim=3):
        self.encoding_dim = encoding_dim  # Setting the coding dimension.
        r = lambda: np.random.randint(1, 3)
        self.x = np.array([[r(), r(), r()] for _ in range(1000)])  # create data to pass in. The input.
        print(self.x)

    def _encoder(self):  # From starting info, compressed into the few abstract features.
        inputs = Input(shape=(self.x[0].shape))
        encoded = Dense(self.encoding_dim, activation='relu')(inputs)
        
        model = Model(inputs, encoded)
        self.encoder = model
        
        return model

    def _decoder(self):  # From the compressed abstract features, expanded back up to a full size.
        inputs = Input(shape=(self.encoding_dim,))
        decoded = Dense(3)(inputs)
        
        model = Model(inputs, decoded)
        self.decoder = model
        
        return model

    def encoder_decoder(self):  # Combine the two separate models to make the whole.
        ec = self._encoder()
        dc = self._decoder()

        inputs = Input(shape=self.x[0].shape)
        ec_out = ec(inputs)
        dc_out = dc(ec_out)
        model = Model(inputs, dc_out)

        self.model = model
        return model

    def fit(self, batch_size=10, epochs=300):  # Use 'Stochastic Gradient Descent' to optimize our weights.
        self.model.compile(optimizer='sgd', loss='mse')  # sgd = Stochastic Gradient Descent. mse = Mean Squared Error.
        log_dir = './log/'   # Log directory. Keeps a record of training progress.
        tbCallBack = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)
        self.model.fit(self.x, self.x,  # Input and target output are the same (self.x)
                       epochs=epochs,  # 'n' epochs = 'n' forward+backward passes over the entire data set.
                       batch_size=batch_size,  # Many batches make up the whole training set.
                       # NOTE: batch_size=10, epochs=300 as stated on line 50.
                       callbacks=[tbCallBack])  # Save the overall loss after each epoch (to visualize progress)

    def save(self):  # To save the weights of the model. So you can reuse a trained model in the future.
        if not os.path.exists(r'./weights'):
            os.mkdir(r'./weights')
        else:
            self.encoder.save(r'./weights/encoder_weights.h5')
            self.decoder.save(r'./weights/decoder_weights.h5')
            self.model.save(r'./weights/ae_weights.h5')


if __name__ == '__main__':
    ae = AutoEncoder(encoding_dim=2)  # create an instance of our AutoEncoder.
    # encoding_dim=2  means that the AutoEncoder will take the input data and represent it as a 2D vector.
    ae.encoder_decoder()  # calling the encoder_decoder function which is initializing the encoder and decoder models.
    ae.fit(batch_size=50, epochs=1)  # fit data set (input) to the Neural Network. 50 items at a time, 300 times total