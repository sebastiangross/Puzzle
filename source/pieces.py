__author__ = 'Herakles II'


import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import misc


from source import config
from source.corners import detectionViaBorder as corner_detection


def create_sides_info(sides):
    res = {}
    res['sides'] = sides
    res['types'] = np.array([np.nan if s in (None, 0) else s.type for s in sides])
    res['nan'] = np.isnan(res['types'])
    res['to_test'] = np.where(np.logical_and(np.logical_not(res['nan']), res['types']!=0))[0]
    res['border'] = np.array([(s == 0) for s in sides])
    return res

class Piece:
    def __init__(self, ID):
        self.id = ID
        self.sides = [Side(parent=self), Side(parent=self), Side(parent=self), Side(parent=self)]
        self.types = None
        self.borders = None
        self.kind = None

    def fit(self, sides, orient, to_test=None):
        if to_test is None:
            to_test = [i for i in range(4) if isinstance(sides[i], Side)]
        fit = np.ones(4)
        for s in to_test:
            fit[s] = self.sides[(s+orient)%4].test_fit(sides[s])
        return np.prod(fit)

    def show(self, orient, ax=None):
        dir = self.corners[::-1][orient-2]-self.corners[::-1][orient-1]
        angle = np.arctan2(dir[1], dir[0])
        img = misc.imrotate(self.img, 180*angle/np.pi)
        ind_alpha = np.where(img[:, :, -1] > 120)
        img = img[np.min(ind_alpha[0]):np.max(ind_alpha[0]), np.min(ind_alpha[1]):np.max(ind_alpha[1]), :]
        if not ax is None:
            plt.sca(ax)
        plt.imshow(img)
        if ax is None:
            plt.show()
        else:
            for spine in ax.spines.values():
                spine.set_edgecolor('white')

    def show_sides(self):
        f = plt.figure()
        for i in range(4):
            ax = f.add_subplot(2, 2, i+1)
            self.sides[i].show_img(ax)
        plt.suptitle('This is piece %s'%self.id)
        plt.show()

    def read_from_img(self, im, corners=None):
        if corners is None:
            corners = corner_detection.find_corners(im)

        angles = np.arctan2(corners[:, 1]-im.shape[1]/2, corners[:, 0]-im.shape[0]/2)
        corners_ord = corners[np.argsort(angles)[::-1], :]

        for p in range(4)[::-1]:
            self.sides[p].read_from_piece_img(im, corners_ord[p-1], corners_ord[p])

        self.corners = corners_ord
        self.img = im

        self.types = np.array([s.type for s in self.sides])
        self.borders = (self.types == 0)
        count_edges = np.sum(self.borders)
        if count_edges == 2:
            self.kind = 'corner'
        elif count_edges == 1:
            self.kind = 'edge'
        elif count_edges == 0:
            self.kind = 'inner'
        else:
            raise Exception('Piece must not have more than two edges!')

    def get_side(self, offset, side_int):
        return self.sides[(offset+side_int)%4]

class Side:
    def __init__(self, length=None, img=np.zeros((config.SIDE_RESOLUTION, config.SIDE_RESOLUTION, 3)), type=None,
                 parent=None):
        self.length = length
        self.img = img
        self.type = type
        self.parent = parent

    def read_from_piece_img(self, piece_img, c1, c2, bordersize=0.2):
        border = int(bordersize*np.max(piece_img.shape))
        piece_img_large = cv2.copyMakeBorder(piece_img, top=border, bottom=border,
                                             left=border, right=border,
                                    borderType=cv2.BORDER_CONSTANT, value=0)
        p1 = c1 + border
        p2 = c2 + border
        mid = np.array([piece_img_large.shape[0]/2, piece_img_large.shape[1]/2])
        from_dir = p2-p1
        self.length = np.linalg.norm(from_dir, ord=2)

        angle = - np.arctan2(from_dir[1], from_dir[0]) + np.pi/4
        rotated = misc.imrotate(piece_img_large, 180*angle/np.pi)

        rot_mat = np.array([[np.cos(angle), -np.sin(angle)],  [np.sin(angle), np.cos(angle)]])
        p1_rot = np.array(np.dot(rot_mat, p1-mid)+mid, dtype=np.int)
        p2_rot = np.array(np.dot(rot_mat, p2-mid)+mid, dtype=np.int)

        shaped = rotated[p1_rot[0]:p2_rot[0], p1_rot[1]:p2_rot[1], :]
        self.img = misc.imresize(shaped, (config.SIDE_RESOLUTION, config.SIDE_RESOLUTION))
        img_alpha = self.img[:, :, -1]
        self.img_alpha = (cv2.blur(img_alpha, (5, 5)) > 120).astype(np.int)

        self.get_border_ind()
        self.get_colors()
        self.clf_type()

    def get_colors(self):
        border_rgb = self.img[self.border_ind[:, 0], self.border_ind[:, 1], :3]
        self.color = cv2.resize(border_rgb[None, :, :], (config.SIDE_RESOLUTION, 1))[0]

    def get_border_ind(self):
        d = np.zeros_like(self.img_alpha)
        d[:, :-1] += np.abs(self.img_alpha[:, :-1] - self.img_alpha[:, 1:])
        d[1:, :] += np.abs(self.img_alpha[:-1, :] - self.img_alpha[1:, :])
        border_all = zip(*np.where(d > 0))
        l = len(border_all)

        border_ind = np.nan*np.zeros(shape=(l, 2))

        end = np.sum(self.img_alpha.shape)
        n = [0, 0]
        for i in range(l):
            dist = np.abs(np.array(border_all)-n).sum(axis=1)
            best_ind = np.argmin(dist)
            if dist[best_ind] <= end - np.sum(n):
                n = border_all[best_ind]
                border_ind[i, :] = n
                border_all.remove(n)
            else:
                break
        mask = np.isfinite(border_ind).any(axis=1)
        self.border_ind = border_ind[mask, :].astype(np.int_)

    def clf_type(self):
        s = np.true_divide(self.border_ind[:, 0] - self.border_ind[:, 1] + config.SIDE_RESOLUTION, 2*config.SIDE_RESOLUTION)
        if len(s) > 0:
            low = 0.5 - s.min()
            high = s.max() - 0.5
            if (low >= config.type_thres) or (high >= config.type_thres):
                if low > high:
                    self.type = 1
                else:
                    self.type = -1
            else:
                self.type = 0
        else:
            self.type = 0

    def show_img(self, ax=None, rev=False):
        if not ax is None:
            plt.sca(ax)
        plt.axis("off")
        if rev:
            plt.imshow(self.img[::-1, ::-1, :])
        else:
            plt.imshow(self.img)
        plt.title('length: %.2f, type: %s'%(self.length, self.type))
        if ax is None:
            plt.show()

    def show_color(self, ax=None, rev=False):
        if not ax is None:
            plt.sca(ax)
        clr = np.zeros((4, self.color.shape[0], 3))
        clr[0, :, :] = self.color
        for i in range(3):
            clr[i+1, :, i] = self.color[:, i]
        if rev:
            plt.imshow(clr[:, ::-1, :])
        else:
            plt.imshow(clr)

        if ax is None:
            plt.show()

    def show(self, ax=None):
        if not ax is None:
            if not isinstance(ax, list):
                ax = [ax, None]
            show = False
        else:
            f, ax = plt.subplots(2, 1)
            show = True
        self.show_img(ax[0])
        self.show_color(ax[1])

        if show:
            plt.show()

    def test_fit(self, new_side, show=False):
        """
        Test the fit with a given side. Whereas no fit is return as 0. And a perfect fit is return as 1.
        """

        # Test if type are different
        if self.type == 0:
            if new_side is 0:
                return 1.
            else:
                return 0.
        elif self.type + new_side.type != 0:
            return 0.
        else:
            kpis = self.test_fit_kpi(new_side, show=show)
            return max(0., 1. - np.dot(config.fit_weights, kpis))

    def test_fit_kpi(self, new_side, show=False):
        """
        Test the fit with a given side. Whereas no fit is return as 0. And a perfect fit is return as 1.
        """

        # Test if type are different
        len_diff = abs(np.log(self.length) - np.log(new_side.length))

        together = self.img_alpha + new_side.img_alpha[::-1, ::-1]
        overlap = (together > 1).mean()
        empty = (together == 0).mean()
        color_diff = np.linalg.norm(self.color.astype(np.int_)-new_side.color[::-1, :].astype(np.int_))/255/config.SIDE_RESOLUTION

        if show:
            ax1 = plt.subplot2grid((4,3), (0, 0), rowspan=2)
            self.show_img(ax1)
            ax2 = plt.subplot2grid((4,3), (0, 1), rowspan=2)
            new_side.show_img(ax2, rev=True)
            ax3 = plt.subplot2grid((4,3), (0, 2), rowspan=2)
            plt.imshow(together)
            ax4 = plt.subplot2grid((4,3), (2, 0), colspan=3)
            self.show_color(ax4)
            ax5 = plt.subplot2grid((4,3), (3, 0), colspan=3)
            new_side.show_color(ax5, rev=True)
            plt.show()

        return (len_diff, overlap, empty, color_diff)