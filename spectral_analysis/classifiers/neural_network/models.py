from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import History
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D, Input, concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical

from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
from kerastuner import HyperModel

import time



def build_cnn(hp, input_length):
    model = Sequential()

    model.add(Conv1D(filters=hp.Int('conv_filters', min_value=32, max_value=512, step=32),
                     kernel_size=hp.Int('kernel_size', min_value=2, max_value=32, step=2),
                     activation='relu',
                     input_shape=(input_length, 1)))

    for i in range(hp.Int('n_layers', 1, 4)):
        model.add(Conv1D(filters=hp.Int(f'conv_{i}_filters', min_value=32, max_value=512, step=32),
                         kernel_size=3,
                         activation='relu'))
                            
    model.add(Dropout(0.5))
    # model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))

    return model

def build_mlp(input_shape):
  model = Sequential()

  model.add(Dense(256, input_dim=input_shape, activation='relu', kernel_initializer='he_uniform'))
  model.add(Dense(256, input_dim=256, activation='relu', kernel_initializer='he_uniform'))

  return model

def build_models(hp, input_shapes, n_classes):
    cnn = build_cnn(input_length=input_shapes['fluxes'], hp=hp)
    mlp = build_mlp(input_shape=input_shapes['source_info'])
    
    combined = concatenate([cnn.output, mlp.output])

    final_classifier = Dense(128, activation="relu")(combined)
    final_classifier = Dense(n_classes, activation="softmax")(final_classifier)
    
    model = Model(inputs=[mlp.input, cnn.input], outputs=final_classifier)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    
    return model

class HyperSpaceModel(HyperModel):
    def __init__(self, n_classes, input_shapes):
        self.n_classes = n_classes
        self.input_shapes = input_shapes

    def build(self, hp):
        cnn = build_cnn(input_length=self.input_shapes['fluxes'], hp=hp)
        mlp = build_mlp(input_shape=self.input_shapes['source_info'])
        
        combined = concatenate([cnn.output, mlp.output])

        final_classifier = Dense(128, activation="relu")(combined)
        final_classifier = Dense(self.n_classes, activation="softmax")(final_classifier)
        
        model = Model(inputs=[mlp.input, cnn.input], outputs=final_classifier)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        return model
