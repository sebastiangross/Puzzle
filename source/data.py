__author__ = 'Herakles II'

from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
from scipy import misc
import multiprocessing
from joblib import Parallel, delayed
import time
import os
import pickle

from source import pieces, config

def load_Board(name, path=config.RESULTS_PATH):
        file = open(path + os.sep + name + '.obj', 'rb')
        return pickle.load(file)

def load_all(name, parallel=False, path=config.DATA_PATH):
    file = path + os.sep + name + os.sep
    print('Loading data from: %s'%file)
    IDs = [f[:-4] for f in listdir(file) if isfile(join(file, f)) and f[-4:] == '.png' and not f == 'total.png']

    if 'corners.csv' in listdir(file):
        corners = pd.read_csv(file + 'corners.csv', index_col=0)
        print('Found additional corner information.')
    else:
        corners = None

    if parallel:
        # # Use this for multi-thread piece import
        n_core = multiprocessing.cpu_count()
        print( 'Loading pieces with %s threads'%n_core)
        ptm = time.time()
        pcs_list = Parallel(n_jobs=n_core)(delayed(load_piece)(i, file) for i in IDs)
        pcs = {pc.id: pc for pc in pcs_list}
        print( 'Total time for loading pieces: %s'%(time.time()-ptm))
    else:
        pcs = {}
        for ID in IDs:
            pcs[ID] = load_piece(ID, file, corners=corners)
    return pcs

def load_piece(ID, path, corners=None):
    im = misc.imread(path + '%s.png'%ID)

    piece = pieces.Piece(ID)

    if corners is None:
        piece.read_from_img(im)
    else:
        c_str = [t.replace('(', '').replace(')', '').split(',') for t in corners.loc[ID]]
        corners_pc = np.array(c_str, dtype=np.int)
        piece.read_from_img(im, corners=corners_pc)

    print( 'Loaded piece %s'%ID)
    return piece