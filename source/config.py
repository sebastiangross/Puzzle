__author__ = 'Herakles II'

import numpy as np
import os
import logging

# side fit
fit_thres = 1 - 0.04
fit_weights = np.array([0.1,  0.1,  0.1,  0.7])

type_thres = 0.05

SIDE_RESOLUTION = 100

search_depth = 1

# data stuff
DATA_PATH = 'data' + os.sep
RESULTS_PATH = 'results' + os.sep

def set_up_logging(name, path=RESULTS_PATH):
    logger = logging.getLogger('puzzle')
    logger.setLevel(logging.INFO)

    # create a file handler
    handler = logging.FileHandler(path + os.sep + name + '.log')
    handler.setLevel(logging.INFO)

    stderr_log_handler = logging.StreamHandler()
    logger.addHandler(stderr_log_handler)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)
    return logger


# DSC01132_10x15: 0.0718698504852/n weights:[ 0.84835836  0.01219194  0.02110715  0.11834256]
# DSC01132_15x30: 0.116710916731/n weights:[  2.73708335e-04   9.54704942e-02   1.07614321e-01   7.96641476e-01]
# IMG_1899_10x15: 0.0644107367057/n weights:[ 0.00207125  0.0228707   0.02249063  0.95256743]
# Bild_3_10x15: 0.121402396412/n weights:[ 0.00968244  0.04954464  0.08011702  0.8606559 ]