__author__ = 'Herakles II'

import numpy as np


from source.corners import detectionViaBorder as viaBorder
#from source.corners import detectionViaNN as viaNN
#from source.corners import detectionViaCNN as viaCNN
from source.corners import helper

if __name__ == '__main__':
    NTrain = 1000
    NTest = 100
    img_size = 128


    #X, y = helper.create_test_set(NTrain, img_size=img_size)
    #net = viaCNN.CNN(img_size, 8, 255)
    #net = viaNN.NN(img_size, 8, 255)
    #net.fit(X, y)

    X, y = helper.create_test_set(NTest, img_size=img_size)
    y_pred = viaBorder.harris(X)

    #y_pred = net.predict(X)
    #print(net.score(X, y))

    #y_pred = np.zeros(shape=(NTest, 4, 2))
    #for n in range(NTest):
    #    y_pred[n, :, :] = viaBorder.find_corners_angles(X[n, :, :], show=False)
    y_true = y.reshape((NTest, 4, 2))
    scrs = helper.score(y_true, y_pred, img_size=img_size)
    helper.showResults(X, y, y_pred, scrs)
    print(y_pred)
    print('via border %s'%scrs.mean())