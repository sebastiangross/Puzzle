#from lasagne import layers
#from lasagne.updates import nesterov_momentum
#from nolearn.lasagne import NeuralNet

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras import backend as K
import theano

import numpy as np
import theano

from source.corners import helper

def float32(k):
    return np.cast['float32'](k)

class NN:
    def __init__(self, img_size, output_num_units, maxValue):
        self.img_size = img_size
        self.input_num_units = img_size**2
        self.output_num_units = output_num_units
        self.maxValue = maxValue

        # # lasange + nolearn
        #self.net = NeuralNet(
        #        layers=[  # three layers: one hidden layer
        #            ('input', layers.InputLayer),
        #            ('hidden', layers.DenseLayer),
        #            ('output', layers.DenseLayer),
        #        ],
        #        # layer parameters:
        #        input_shape=(None, self.input_num_units),  # input pixels per batch
        #        hidden_num_units=100,  # number of units in hidden layer
        #        output_nonlinearity=None,  # output layer uses identity function
        #        output_num_units=output_num_units,  # target values#
        #
        #        # optimization method:
        #        update=nesterov_momentum,
        #        update_learning_rate=theano.shared(float32(0.03)),
        #        update_momentum=theano.shared(float32(0.9)),
        #
        #        regression=True,  # flag to indicate we're dealing with regression problem
        #        max_epochs=200,  # we want to train this many epochs
        #        verbose=1,
        #    )

        # # keras
        model = Sequential()
        model.add(Dense(200, input_dim=self.input_num_units, activation=K.tanh))
        model.add(Dense(output_num_units))

        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='mean_squared_error', optimizer=sgd)
        self.net = model

    def _scale(self, X=None, y=None):
        if X is None:
            XOut = None
        else:
            N = X.shape[0]
            XOut = (X / self.maxValue).reshape((N, self.input_num_units)).astype(np.float32)
        if y is None:
            yOut = None
        else:
            N = y.shape[0]
            yOut = y.reshape((N, 8)).astype(np.float32)/self.img_size
        return XOut, yOut

    def fit(self, X, y):
        XScaled, yScaled = self._scale(X, y)
        self.net.fit(XScaled, yScaled)

    def predict(self, X):
        XScaled, _ = self._scale(X)
        yPred = self.net.predict(XScaled)
        return yPred.reshape((yPred.shape[0], 4, 2)) * self.img_size

    def score(self, X, y):
        yPred = self.predict(X)
        return helper.score(y, yPred, self.img_size)