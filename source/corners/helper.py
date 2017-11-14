import numpy as np
from matplotlib import pyplot as plt

from source import create_random_puzzle


def create_test_set(N, img_size=20):
    X = np.zeros(shape=(N, img_size, img_size), dtype=np.uint8)
    y = np.zeros(shape=(N, 4, 2))

    for i in range(N):
        img, c = create_random_puzzle.create_piece(size=img_size, show=False, scale=0.05, rotate=True)
        X[i, :, :] = img
        y[i, :, :] = c
    return X, y


def save_test_set(X, y):
    N = X.shape[0]
    img_size = X.shape[1]
    np.savez_compressed('test_images_%s' % N, y=y,
                        X=X.reshape((N, img_size * img_size)))


def load_test_set(name):
    npzfile = np.load(name + '.npz')
    N, img_size_sq = npzfile['X'].shape
    img_size = np.int(np.sqrt(img_size_sq))
    X = npzfile['X'].reshape((N, img_size, img_size))
    return X, npzfile['y']


def split_sets(X, y, ratio=0.8):
    if not X.shape[0] is y.shape[0]:
        raise Exception('X and y must have same length!')
    N = X.shape[0]
    N_train = int(N*ratio)
    return (X[:N_train], y[:N_train]), (X[N_train:], y[N_train:])


def score(Y_true, Y_pred, img_size):
    if len(Y_true.shape) == 2:
        Y_true = Y_true[None, :]
    if len(Y_pred.shape) == 2:
        Y_pred = Y_pred[None, :]
    if Y_true.shape[0] != Y_pred.shape[0]:
        raise Exception('Y1 and Y2 must have same length')

    s = np.zeros(Y_true.shape[0])

    for i, y2 in enumerate(Y_pred):
        y1 = Y_true[i]
        dist = np.sqrt(np.sum((y1[None, :, :] - y2[:, None, :])**2, axis=2))
        s[i] = dist.min(axis=0).mean()
    return s/img_size


def showResults(X, y, yPred, scores, nShow=4):
    order = np.argsort(scores)
    toShow = list(order[:nShow]) +list(order[-nShow:])

    f, axs = plt.subplots(2, 4)

    for i in range(2*nShow):
        j = toShow[i]
        axs.flat[i].imshow(X[j])
        axs.flat[i].scatter(yPred[j, :, 1], yPred[j, :, 0])
    plt.show()