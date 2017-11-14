__author__ = 'Herakles II'

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.interpolate as si
import cv2
import os
from scipy import misc
from copy import deepcopy

def rotate_matrix(angle):
    return np.array([[np.cos(angle), -np.sin(angle)],  [np.sin(angle), np.cos(angle)]])

def bspline(points):
    x = points[:,0]
    y = points[:,1]

    t = range(len(points))
    ipl_t = np.linspace(0.0, len(points) - 1, 100)

    x_tup = si.splrep(t, x, k=2)
    y_tup = si.splrep(t, y, k=2)

    x_list = list(x_tup)
    xl = x.tolist()
    x_list[1] = xl + [0.0, 0.0, 0.0, 0.0]

    y_list = list(y_tup)
    yl = y.tolist()
    y_list[1] = yl + [0.0, 0.0, 0.0, 0.0]

    x_i = si.splev(ipl_t, x_list)
    y_i = si.splev(ipl_t, y_list)
    return np.column_stack((x_i, y_i))

def create_random_spline(p1, p2, rc=None, border=True):
    x = p2 - p1
    if rc is None:
        if border:
            rc = np.random.choice([-1, 0, 1], p=[0.45, 0.1, 0.45])
        else:
            rc = np.random.choice([-1, 1])

    if rc == 0:
        spline = np.array([p1, p2], dtype=np.int)
    else:
        y = np.dot(rotate_matrix(-rc*np.pi/2), x)
        mid_x = np.random.uniform(low=0.4, high=0.6)
        mid_y = np.random.uniform(low=-0.05, high=0.05, size=3)
        radius = np.random.uniform(low=0.12, high=0.15)
        offset = np.random.uniform(low=-0.05, high=0.05, size=2)
        height = np.random.uniform(low=-0., high=0.05)
        width = np.random.uniform(low=0., high=radius/2, size=2)
        angle = np.random.uniform(low=-0.15, high=0.15)*np.pi

        mid = p1 + mid_x*x + mid_y[1]*y
        center = mid + np.dot(rotate_matrix(angle), y)*(radius+height)

        points = np.row_stack((p1,
                               (p1 + mid)/2 + y*mid_y[0],
                               mid - x*(width[0]),
                               center + -x*(radius+offset[0]),
                               center + (-x+y)/2*(radius+offset[0]),
                               center + (x+y)/2*(radius+offset[1]),
                               center + x*(radius+offset[1]),
                               mid + x*(width[1]),
                               (p2 + mid)/2 + y*mid_y[1],
                               p2))
        spline = bspline(points)
        spline = np.array(spline, dtype=np.int)
    return spline

def create_piece(size=1000, show=False, rotate=True, scale=0.05):
    #function will create a random grayscale jigsaw piece
    corners = np.random.normal(scale=scale*size, size=8)
    corners = corners.reshape((4,2))
    corners += [size/4, size/4]
    corners[[1,2], :] += [size/2, 0]
    corners[[2,3], :] += [0, size/2]
    corners = np.array(corners, dtype=np.int)

    splines = []

    for i in range(4)[::-1]:
        p1 = corners[i]
        p2 = corners[i-1]

        spline = create_random_spline(p1, p2)
        splines.append(spline)
    splines = np.row_stack(tuple(splines))

    if rotate:
        angle = 2*np.pi*np.random.rand()
        mid = (size/2, size/2)

        rot_mat = rotate_matrix(angle)
        corners = np.array(np.dot(corners-mid, rot_mat)+mid, dtype=np.int)
        splines = np.array(np.dot(splines-mid, rot_mat)+mid, dtype=np.int)

    img_empty = np.zeros(shape=(size, size))
    img = cv2.fillPoly(img_empty, [splines], color=255)

    if show:
        plt.imshow(img)
        plt.show()
    return img, corners[:, ::-1]

def create_random_side_pair(size=500):
    fit = np.random.choice([False, True], p=[0.95, 0.05])
    if fit:
        s1 = create_random_side(size)
        s2 = (255-s1)[::-1, ::-1]
    else:
        s1 = create_random_side(size)
        s2 = create_random_side(size)
    return s1, s2, fit

def create_random_side(size=500):
    img_empty = np.zeros(shape=(size, size))
    t = np.random.choice([-1, 0, 1], p=[0.45, 0.1, 0.45])
    sp = create_random_spline(p1=np.zeros(2), p2=(size-1)*np.ones(2), rc=t)
    splines = np.row_stack((sp, np.array([0, size-1])))
    img = cv2.fillPoly(img_empty, [splines], color=255)
    return img, t

def create_from_im(path, file, size, std_corner=0., rescale=None, rotate=False, border=0.1):
    # let's begin
    n_row, n_col = size
    dir_out = '%s_%sx%s'%(file[:-4], size[0], size[1]) + os.sep
    path_out = path + os.sep + dir_out
    os.makedirs(path_out)
    whole_im = misc.imread(path + os.sep + file)
    if not rescale is None:
        whole_im = cv2.resize(whole_im, rescale)

    whole_alpha = np.concatenate((whole_im,
                                  255*np.ones_like(whole_im[:, :, 0, None])),
                                 axis=2)

    pieces_corner = pd.DataFrame(index=['c_ul', 'c_ur', 'c_lr', 'c_ll'])

    x = np.linspace(0, whole_im.shape[0], num=n_row+1)
    y = np.linspace(0, whole_im.shape[1], num=n_col+1)

    xx, yy = np.meshgrid(x, y)
    grid = np.concatenate((xx.T[None, :, :], yy.T[None, :, :]), axis=0)

    if std_corner > 0:
        grid[0, 1:-1, :] += np.random.normal(scale=std_corner, size=(n_row-1, n_col+1))
        grid[1, :, 1:-1] += np.random.normal(scale=std_corner, size=(n_row+1, n_col-1))

    grid = grid.astype(np.int)

    for row in range(n_row):
        for col in range(n_col):
            p_u = grid[:, row+1, col]
            p_c = grid[:, row+1, col+1]
            p_l = grid[:, row, col+1]

            if row == n_row-1:
                spline_right = np.array([p_c, p_u])
            else:
                spline_right = create_random_spline(p_c, p_u, border=False)
            if col == n_col-1:
                spline_lower = np.array([p_l, p_c])
            else:
                spline_lower = create_random_spline(p_l, p_c, border=False)

            x_max = spline_right[:, 0].max()
            y_max = spline_lower[:, 1].max()

            spline0 = np.array([[p_u[0]-p_u[1], 0],
                                [x_max, 0],
                                [x_max, y_max],
                                [0, y_max],
                                [0, p_l[1]-p_l[0]]], dtype=np.int)

            splines = [spline_lower, spline_right, spline0]

            piece_alpha = cv2.fillPoly(whole_alpha[:(x_max+1), :(y_max+1), -1].astype(np.int),
                                       [np.row_stack(tuple(splines))[:, ::-1]], color=0)

            piece_im = np.concatenate((whole_alpha[:(x_max+1), :(y_max+1), :-1],
                                       piece_alpha[:, :, None]),
                                      axis=2)
            mask = (piece_alpha > 0)
            whole_alpha[:piece_alpha.shape[0], :piece_alpha.shape[1], -1][mask] = 0

            x_min = np.argmax(mask.any(axis=1))
            y_min = np.argmax(mask.any(axis=0))

            piece_im = piece_im[x_min:, y_min:, :]

            bordersize = int(piece_im.shape[0]*border)
            piece_border = cv2.copyMakeBorder(piece_im, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize,
                                        borderType=cv2.BORDER_CONSTANT, value=0)

            c = np.array([grid[:, row, col], p_u, p_c, p_l])
            c_shift = c + bordersize - [x_min, y_min]

            if rotate:
                angle = 2*np.pi*np.random.rand()
                mid = (piece_border.shape[0]/2, piece_border.shape[1]/2)

                rot_mat = rotate_matrix(angle)
                c_shift = np.array(np.dot(c_shift-mid, rot_mat)+mid, dtype=np.int)
                piece_border = misc.imrotate(piece_border, -180*angle/np.pi)

            name = 'p_%s_%s'%(row, col)
            misc.imsave(path_out + '%s.png'%name, piece_border)
            pieces_corner[name] = zip(*c_shift.T)

            whole_copy = cv2.polylines(whole_im, [np.row_stack(tuple([spline_lower, spline_right]))[:, ::-1]],
                                       isClosed=False, color=(0, 0, 0))

    (pieces_corner.T).to_csv(path_out + 'corners.csv')
    misc.imsave(path_out + 'total.png', whole_im)