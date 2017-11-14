__author__ = 'Herakles II'

import os
from source.create_random_puzzle import create_from_im

dirname, _ = os.path.split(os.path.abspath(__file__))

if __name__ == '__main__':

    path = dirname + os.sep + 'img'
    file = 'Bild_3.JPG'
    #file = 'DSC01132.JPG'

    std_corner = 4 #standard deviation of corners
    size = (3, 5)
    rescale = None#(2000, 1200)#

    create_from_im(path, file, size, std_corner=std_corner, rescale=rescale, rotate=True, border=0.3)
