import numpy as np
from keras.models import Model, Sequential  # basic class for specifying and training a neural network
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten, Reshape
from keras.utils import np_utils  # utilities for one-hot encoding of ground truth values

from source.corners import helper

def float32(k):
    return np.cast['float32'](k)

class CNN:
    def __init__(self, img_size, output_num_units, maxValue):
        self.img_size = img_size
        self.input_num_units = img_size
        self.output_num_units = output_num_units
        self.maxValue = maxValue

        self.net = Sequential()
        self.net.add(Convolution2D(filters=32, kernel_size=(3, 3), activation='sigmoid',
                                   input_shape=(1, img_size, img_size)))
        self.net.add(MaxPooling2D(pool_size=(2, 2)))
        #self.net.add(Dropout(0.1))
        self.net.add(Convolution2D(filters=64, kernel_size=(2, 2), activation='sigmoid'))
        self.net.add(MaxPooling2D(pool_size=(2, 2)))
        #self.net.add(Dropout(0.2))
        self.net.add(Convolution2D(filters=128, kernel_size=(2, 2), activation='sigmoid'))
        self.net.add(MaxPooling2D(pool_size=(2, 2)))
        #self.net.add(Dropout(0.3))
        self.net.add(Flatten())
        self.net.add(Dense(500, activation='sigmoid'))#, activation='relu'))
        #self.net.add(Dropout(0.5))
        self.net.add(Dense(500, activation='sigmoid'))
        self.net.add(Dense(output_num_units, activation='linear'))

        self.net.compile(loss='mean_squared_error',
                      optimizer='Nadam',
                      metrics=['accuracy'])  # reporting the accuracy

        # es hat nicht funktioniert.
        # Ergebnis reagiert nicht auf Input, warum??

    def _scale(self, X=None, y=None):
        if X is None:
            XOut = None
        else:
            N = X.shape[0]
            XOut = X.astype(np.float32)[:, None, :, :]/ self.maxValue
        if y is None:
            yOut = None
        else:
            N = y.shape[0]
            yOut = (2 * y.astype(np.float32)/self.img_size - 1).reshape((N, 8))
        return XOut, yOut

    def fit(self, X, y):
        XScaled, yScaled = self._scale(X, y)

        self.net.fit(XScaled, yScaled,  # Train the model using the training set...
                     epochs=100, verbose=1, validation_split=0.1)

    def predict(self, X):
        XScaled, _ = self._scale(X)
        yPred = self.net.predict(XScaled)
        return (yPred.reshape((yPred.shape[0], 4, 2)) + 1) * self.img_size/2

    def score(self, X, y):
        yPred = self.predict(X)
        return helper.score(y, yPred, self.img_size)