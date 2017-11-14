__author__ = 'Herakles II'


import itertools
import pandas as pd
from joblib import Parallel, delayed
import time
import numpy as np
import multiprocessing

from source import data

dir_check = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

def partition(lst, n):
    q, r = divmod(len(lst), n)
    indices = [q*i + min(i, r) for i in xrange(n+1)]
    return [lst[indices[i]:indices[i+1]] for i in xrange(n)]

def get_test_fit_many(index, sds_m, sds_f):
    res = len(index)*[None]
    for i in range(len(index)):
        key = index[i]
        r = get_test_fit(key, sds_m[key[0]], sds_f[key[1]])
        res[i] = r
    return res

def get_test_fit(key, s1, s2):
    X = s1.test_fit_kpi(s2)
    m, f = key
    d = dir_check[m[2]]
    if (m[0] + d[0] == f[0]) and (m[1] + d[1] == f[1]):
        y = 1
    else:
        y = 0
    #print 'Done with pair: %sx%s'%key
    return (key, X, y)

def split_key(key, i):
    str = key.split('_')
    return (int(str[1]), int(str[2]), i)

def create_kpis(file):
    pcs = data.load_all(file, parallel=False)

    sds = {split_key(key, i): pcs[key].sides[i] for i in range(4) for key in pcs}

    sds_m = {k: sds[k] for k in sds if sds[k].type == 1}
    sds_f = {k: sds[k] for k in sds if sds[k].type == -1}

    index = list(itertools.product(sds_m, sds_f))

    n_core = multiprocessing.cpu_count()
    print 'Testing tuples (%s) with %s threads'%(len(list(index)), n_core)
    ptm = time.time()
    index_meta = partition(index, n_core)
    X_listlist = Parallel(n_jobs=n_core)(delayed(get_test_fit_many)(key, sds_m, sds_f) for key in index_meta)
    print 'Total time for testing fits: %s'%(time.time()-ptm)

    X_list = itertools.chain.from_iterable(X_listlist)
    Xy = pd.DataFrame.from_dict({x[0]: {'X1': x[1][0], 'X2': x[1][1], 'X3': x[1][2], 'X4': x[1][3], 'y': x[2]} for x in X_list}, orient='index')
    Xy.to_csv(file + '_kpis.csv')

def analyse_kpis(file):
    kpis = pd.read_csv(file)#, converters ={'X': convert_X})
    columns = list(kpis.columns)
    columns[:2] = ['id1', 'id2']
    kpis.columns = columns

    #kpis = kpis.head(100000)

    y_pivot = kpis.pivot(index='id1', columns='id2', values='y')
    exp = y_pivot.replace(0, np.nan)

    X1_pivot = kpis.pivot(index='id1', columns='id2', values='X1')
    X2_pivot = kpis.pivot(index='id1', columns='id2', values='X2')
    X3_pivot = kpis.pivot(index='id1', columns='id2', values='X3')
    X4_pivot = kpis.pivot(index='id1', columns='id2', values='X4')

    X_pivot = np.concatenate((X1_pivot.values[:, :, None],
                              X2_pivot.values[:, :, None],
                              X3_pivot.values[:, :, None],
                              X4_pivot.values[:, :, None]), axis=2)

    fits = y_pivot.replace(0, np.nan)
    no_fits = (1-y_pivot).replace(0, np.nan)


    def loss(w):
        pred = np.dot(X_pivot, w/w.sum())
        worst_fit = np.nanmax(fits * pred, axis=1)
        best_non_fit = np.nanmin(no_fits * pred, axis=1)
        return np.nanmean(np.maximum(0, worst_fit - best_non_fit))

    w0 = np.array([2, 1, 1, 1], dtype=np.float_)

    from scipy.optimize import minimize

    res = minimize(loss, w0, bounds=4*[(0,None)])
    if res['status'] == 0:
        pred = np.dot(X_pivot, res['x']/res['x'].sum())
        thres = np.nanmax(fits * pred)
        print 'Solution found!/n function value: %s/n weights:%s'%(thres, res['x']/res['x'].sum())
        return True
    else:
        print 'Did not find solution!'
        return False

if __name__ == '__main__':

    #path = 'data/DSC01132_25x40/'
    #path = 'data/IMG_1899_10x15/'
    path = 'DSC01132_10x15'
    #path = 'DSC01132_15x30'
    #path = 'data/Bild_3_10x15/'

    create_kpis(path)

    #file = 'DSC01132_25x40_kpis.csv'
    #file = 'DSC01132_10x15_kpis.csv'
    #file = 'IMG_1899_10x15_kpis.csv'
    #file = 'DSC01132_15x30_kpis.csv'
    #file = 'Bild_3_10x15_kpis.csv'

    analyse_kpis(file)