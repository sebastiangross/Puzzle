__author__ = 'Herakles II'


import cv2
import matplotlib.pyplot as plt
import numpy as np

from source.corners import cartToPolar

def get_gray_img(im):
    # get gray image
    if len(im.shape) > 2:
        if im.shape[2] > 1:
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        else:
            gray = im[:, :, 0]
    else:
        gray = im
    return cv2.GaussianBlur(gray, (7, 7), 0)

def get_border_points(gray, thres):
    mask = (gray > thres)
    points = []
    for ax in range(2):
        g = np.diff(mask, axis=ax)
        ind = np.where(g != 0)
        p = np.array(ind, dtype=np.float_).T
        p[:, ax] += 0.5
        points.append(p)
    points = np.concatenate(points, axis=0)
    return points

def find_corners_kalman(im, thres=120, show=False):
    gray = get_gray_img(im)

    # determine border points
    points = get_border_points(gray, thres)
    num_points = points.shape[0]

    # order border points
    this = 0
    ind_order = [this]
    dist_order = []
    ind_open = range(1, num_points)
    while len(ind_open) > 0:
        dist = np.sum((points[ind_open, :] - points[this, :])**2, axis=1)
        winner = np.argmin(dist)
        dist_order.append(dist[winner])
        this = ind_open[winner]
        ind_order.append(this)
        ind_open.remove(this)

    if show:
        plt.scatter(points[ind_order, 0], points[ind_order, 1], c=range(num_points), cmap='gray')
        plt.show()

    a = 1


def find_corners_angles(im, thres=120, show=False):
    gray = get_gray_img(im)

    # determine border points
    points = get_border_points(gray, thres)
    num_points = points.shape[0]

    # order border points
    this = 0
    ind_order = [this]
    dist_order = []
    ind_open = list(range(1, num_points))
    while len(ind_open) > 0:
        dist = np.sum((points[ind_open, :] - points[this, :])**2, axis=1)
        winner = np.argmin(dist)
        dist_order.append(dist[winner])
        this = ind_open[winner]
        ind_order.append(this)
        ind_open.remove(this)

    if show:
        plt.scatter(points[ind_order, 0], points[ind_order, 1], c=range(num_points), cmap='gray')
        plt.show()

    # moving regression
    half_window = int(num_points/85)
    directions = np.zeros(shape=(num_points, 2))
    ind_order_extended = ind_order[-half_window:] + ind_order + ind_order[:half_window]
    for i in range(num_points):
        p = points[ind_order_extended[i:(i+2*half_window)]]
        p_offset = p - p[half_window, :]
        base = np.concatenate((p_offset[-1, :, None], [[-p_offset[-1, 1]], [p_offset[-1, 0]]]), axis=1)
        base_switch = np.linalg.inv(base)
        q = np.dot(base_switch, (p-p[0, :]).T).T
        lm = np.polyfit(q[:, 0], q[:, 1], deg=1)
        dir = np.dot(base, [1, lm[0]])
        directions[i] = dir/np.linalg.norm(dir, ord=2)

    window = int(num_points/4/6)
    directions_extended = np.concatenate((directions[-window:], directions,
                                          directions[:2*window]), axis=0)
    res = np.zeros(shape=(num_points, 2))
    for i in range(num_points):
        a1 = directions_extended[i:(i+window)]
        a2 = directions_extended[(i+window):(i+2*window)]
        a1_mean = a1.mean(axis=0)
        a1_angle = np.arctan2(a1_mean[1], a1_mean[0])
        a2_mean = a2.mean(axis=0)
        a2_angle = np.arctan2(a2_mean[1], a2_mean[0])
        res[i] = ((a1_angle - a2_angle+ np.pi)%(2*np.pi)-np.pi,
                  a1.std(axis=0).mean() + a2.std(axis=0).mean())

    test = np.logical_and(res[:, 0] < -np.pi/4, np.abs(res[:, 1]) < np.pi/13)
    found = list(np.where(test))

    from sklearn.cluster import KMeans
    clf = KMeans(n_clusters=4)
    clf.fit(list(zip(*found)))
    corners = np.zeros(shape=(4,2))
    for i in range(4):
        floor = np.int(np.floor(clf.cluster_centers_[i]))
        ceil = np.int(np.ceil(clf.cluster_centers_[i]))
        rem = clf.cluster_centers_[i] - floor
        corners[i] = (1-rem)*points[ind_order[floor]] + rem*points[ind_order[ceil]]

    if show:
        f = plt.figure()
        f.add_subplot(3, 1, 1)
        plt.plot(directions)
        f.add_subplot(3, 1, 2)
        plt.plot(res)
        f.add_subplot(3, 1, 3)
        plt.plot(test)
        plt.show()

    return corners


def find_corners(im, thres=120):
    gray = get_gray_img(im)

    mask = (gray > thres)
    origin = np.array([np.mean(np.where(mask.any(axis=1))[0]),
                       np.mean(np.where(mask.any(axis=0))[0])],
                      dtype=np.int)

    polar_grid, r, theta = cartToPolar.reproject_image_into_polar(gray[:, :, None], origin)  # , scale=2000)
    polar_grid = polar_grid[:, :, 0]

    mask = np.array(polar_grid > thres, dtype=np.int)
    grad = np.diff(mask, axis=1)

    # plt.imshow(grad)
    # plt.show()

    neg = find_points(grad, 'neg')
    pos = find_points(grad, 'pos')

    dist = np.zeros(shape=(len(neg), 3))
    for i in range(len(neg)):
        possible = np.where(pos[:, 1] < neg[i, 1])[0]
        if len(possible) > 0:
            d = np.sum((neg[i, :2] - pos[possible, :2]) ** 2, axis=1)
            winner = np.argmin(d)
            dist[i, :] = [possible[winner], d[winner], pos[possible[winner], 2] - neg[i, 2]]
        else:
            dist[i, :] = [np.nan, np.inf, np.inf]

    loss = 1.*dist[:, 1] + 0.001*dist[:, 2]
    arg_sorted = np.argsort(loss)
    best = arg_sorted[:4]

    corners_pol = np.zeros(shape=(4, 2), dtype=np.int)
    try:
        for i in range(4):
            corners_pol[i, :] = (neg[best[i], :2] + pos[int(dist[best[i], 0]), :2]) / 2
    except:
        a=1

    theta_found = theta[corners_pol[:, 1]]
    r_found = r[corners_pol[:, 0]]

    corners_rel = cartToPolar.polar2cart(r_found, theta_found)
    corners = np.array(zip(*corners_rel), dtype=np.int) + origin
    return corners[:, ::-1]


def find_points(g, t):
    if t == 'pos':
        all = g > 0
        r = range(all.shape[1])[::-1]
    else:
        all = g < 0
        r = range(all.shape[1])
    groups = []
    for col in r:
        found = np.where(all[:, col])[0]
        if len(found) > 0:
            most_interesting = np.max(found)
            if len(groups) > 0:
                if most_interesting > groups[-1][-1, 0]:
                    groups.append(np.array([[most_interesting, col]]))
                else:
                    groups[-1] = np.append(groups[-1], [[most_interesting, col]], axis=0)
            else:
                groups.append(np.array([[most_interesting, col]]))
        else:
            continue
    points = np.array([[g[0][0], g[0][1], np.mean(np.diff(g[:4, 1])/np.diff(g[:4, 0]))] for g in groups if (len(g) > 4)])
    return points

def harris(ims):
    count = ims.shape[0]
    y = np.zeros(shape=(count, 4, 2))
    for i in range(count):
        heatMap = cv2.cornerHarris(ims[i], blockSize=5, ksize=3, k=10)
        #plt.imshow(heatMap)
        border_points = get_border_points(ims[i], 120)
        hull = cv2.convexHull(border_points.astype(np.int))
        hull = hull[:, 0, :]
        dist = np.sqrt(np.sum((hull[:, :, None] - hull[:, :, None].T)**2, axis=1))
        dist_max_half = dist.max()/4
        r = list(map(lambda x: heatMap[(x[0] - 2):(x[0] + 2), (x[1] - 2):(x[1] + 2)].min(), zip(*hull.T)))
        p = list(hull[np.argsort(r), :])

        I = [p[0]]
        while len(I) < 4 and len(p) > 1:
            dist = np.sqrt(np.sum((np.array(I) - np.array(p[0]))**2, axis=1))
            if dist.min() >= dist_max_half:
                I.append(p[0])
            p = p[1:]
        if len(I) == 4:
            y[i] = np.array(I)

        #y[i] = hull[w[:4], 0, :]
        #plt.imshow(ims[i])
        #plt.scatter(hull[:, 0, 1], hull[:, 0, 0])
        #plt.show()
    return y