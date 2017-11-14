__author__ = 'Herakles II'

from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import shutil


def split(filename, background=(0, 255, 0), lower_threshold=0.01, upper_threshold=0.9, show=True, path_out='pieces'):
    if os.path.isdir(path_out):
        shutil.rmtree(path_out)
    os.makedirs(path_out)

    img = misc.imread(filename)
    mask = 255*(1-np.all(img == background, axis=2).astype(np.uint8))
    img_alpha = np.concatenate((img, mask[:, :, None]), axis=2)

    if show:
        plt.imshow(img_alpha)
        plt.title('Whole image')
        plt.show()

    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    total_volume = img.shape[0]*img.shape[1]
    count = 0

    for contour in contours:
        volume = cv2.contourArea(contour)
        if (volume < upper_threshold*total_volume) and (volume > lower_threshold*total_volume):
            poly = contour[:, 0, :]

            piece = np.zeros(img_alpha.shape, np.uint8)
            piece = cv2.fillPoly(piece, [poly], color=(0, 0, 0, 255))
            ind = (piece == (0,0,0,255)).all(axis=2)
            for i in range(3):
                piece[:, :, i][ind] = img_alpha[:, :, i][ind]

            piece = piece[poly[:, 1].min():poly[:, 1].max(),
                          poly[:, 0].min():poly[:, 0].max(), :]

            name = 'p_%s.png'%count
            misc.imsave(path_out + os.sep + name, piece)
            count += 1

if __name__ == '__main__':

    file = 'Puzzle_greenscreen.png'
    split(file, show=False)
