__author__ = 'Herakles II'

import pandas as pd
import numpy as np
from copy import deepcopy
import itertools
import matplotlib
import matplotlib.pyplot as plt
import os
import time
import multiprocessing
from joblib import Parallel, delayed

from source import pieces, config, data

def find_fits(brd, pos, depth, pos_mask):
    res = brd.tree_find_fits(pos, depth=depth, pos_mask=pos_mask)
    return (pos, res)

def partition(lst, n):
    q, r = divmod(len(lst), n)
    indices = [q*i + min(i, r) for i in xrange(n+1)]
    return [lst[indices[i]:indices[i+1]] for i in xrange(n)]

def get_test_fit_many(index, sds_m, sds_f):
    res = len(index)*[None]
    for i in range(len(index)):
        key = index[i]
        r = sds_m[key[0]].test_fit(sds_f[key[1]])
        res[i] = (key, r)
    return res

class Board:
    def __init__(self, height=0, width=0, name=''):
        self.name = name
        self.height = height
        self.width = width
        self.fields = pd.DataFrame()
        self.box = {}
        self.fit = 1.
        self.sides_fit = pd.Series()

        self.unused = []
        self.create_fields()

    def test_all_sides(self, parallel=False):
        sds = {(key, i): self.box[key].sides[i] for i in range(4) for key in self.box}

        sds_m = {k: sds[k] for k in sds if sds[k].type == 1}
        sds_f = {k: sds[k] for k in sds if sds[k].type == -1}

        index = list(itertools.product(sds_m, sds_f))

        if parallel:
            n_core = multiprocessing.cpu_count()
            ptm = time.time()
            print('Testing tuples (%s) with %s threads'%(len(list(index)), n_core))
            index_meta = partition(index, n_core)
            X_listlist = Parallel(n_jobs=n_core)(delayed(get_test_fit_many)(key, sds_m, sds_f) for key in index_meta)
            X_list = itertools.chain.from_iterable(X_listlist)
            print('Total time of testing: %s'%(time.time()-ptm))
        else:
            X_list = get_test_fit_many(index, sds_m, sds_f)
        ind_out, values = zip(*X_list)
        self.sides_fit = pd.Series(values, index=pd.MultiIndex.from_tuples(ind_out))

    def copy(self):
        brd = Board(height=self.height, width=self.width, name=self.name+'_copy')
        brd.fields = deepcopy(self.fields)
        brd.sides_fit = self.sides_fit
        brd.fill_box(self.box)
        return brd

    def load_pieces(self):
        pcs = data.load_all(self.name)
        self.fill_box(pcs)
        self.determine_field_size()

    def determine_field_size(self):
        border_pcs = [pc for pc in self.unused if (self.box[pc].kind in ('edge', 'corner'))]
        nBrPcs = len(border_pcs)
        nPcs = len(self.box)

        if np.mod(nBrPcs, 2) > 0:
            raise Exception('Number of border pieces must be dividable by two.')

        c = nBrPcs/4. + 1
        d = np.sqrt(c**2 - nPcs)
        size = np.array((c-d, c+d))
        i_size = np.array(size, dtype=np.int_)

        if np.linalg.norm(size - i_size, ord=1) > 0:
            raise Exception('Number of border pieces and total number of pieces do not match.')
        else:
            print('Puzzle is of size: %s'%size)

        self.height, self.width = i_size
        self.create_fields()

    def obtain_border(self, depth=4):
        print('Try to fill border!')
        kinds = pd.Series({pc: self.box[pc].kind for pc in self.unused})
        edge_pcs = list(kinds.index[kinds == 'edge'])
        corner_pcs = list(kinds.index[kinds == 'corner'])
        border_pcs = edge_pcs + corner_pcs

        pos_mask = (list(itertools.product([0, self.height-1], range(self.width)))
                    + list(itertools.product(range(1, self.height-1), [0, self.width-1])))

        for corner in corner_pcs:
            print('Try new corner...' + corner)
            brd = deepcopy(self)
            for orient in range(4):
                if np.all(np.array([True, False, False, True]) == np.roll(self.box[corner].borders, -orient)):
                    break
            brd.set_piece(corner, pos=(0, 0), orient=orient)
            success = brd.try_to_solve_puzzle_tree(depth=depth, pos_mask=pos_mask)
            if success:
                print('Successfully obtained border!')
                self.fields = brd.fields
                self.determine_unused()
                return True
        print('Unable to obtain border!')
        return False

    def solve_puzzle_single_fields(self, NIter=10000, pos_mask=None):

        for nIter in range(NIter):
            if not len(self.unused) > 0:
                return True
            pot_pos = self.get_potential_positions()
            if not pos_mask is None:
                pot_set = set(pos_mask).intersection(set(pot_pos))
                pot_pos = list(pot_set)
            if not len(pot_pos) > 0:
                return True
            fits = self.try_to_fill_positions(pot_pos)
            if not fits.sum() > 0:
                return False

        print('Reached iteration limit of %s'%NIter)
        return False

    def fill_box(self, pcs):
        if isinstance(pcs, list):
            try:
                self.box = {p.id: p for p in pcs}
            except:
                raise Exception('Box can only contain objects of type pieces.Piece')
        elif isinstance(pcs, dict):
            if not all([isinstance(p, pieces.Piece) for p in pcs.values()]):
                raise Exception('Box can only contain objects of type pieces.Piece')
            self.box = pcs
        else:
            raise Exception('Box must be a dict or list.')
        self.determine_unused()

    def determine_unused(self):
        used = set(self.fields.applymap(lambda x: x[0]).values.flat)
        self.unused = set(self.box.keys()).difference(used)

    def create_fields(self):
        d = {i: {j: (None, 0) for j in range(self.height)} for i in range(self.width)}
        self.fields = pd.DataFrame.from_dict(d, orient='columns')

    def set_piece(self, piece_id, pos, orient, fit=1.):
        if not piece_id in self.box:
            raise Exception('Given piece_id is not contained in box.')
        if not piece_id in self.unused:
            raise Exception('Given piece_id is already used.')

        borders_pc = np.roll(self.box[piece_id].borders, -orient)
        borders_fld = np.array([pos[0] == 0, pos[1] == self.width-1,
                                pos[0] == self.height-1, pos[1] == 0])

        if not all(borders_pc == borders_fld):
            raise Exception('Piece does not fit field position.')

        self.fit *= fit
        self.fields.loc[pos[0], pos[1]] = (piece_id, orient)
        self.unused.remove(piece_id)

    def find_potential_fits(self, pos, sides=None, unused=None):
        if sides is None:
            sides = self.get_neighboring_sides(pos)
        if unused is None:
            unused = self.unused

        pc_kind = {0: 'inner', 1:'edge', 2:'corner'}[sides.count(0)]
        pcs_potential = [pc for pc in unused if self.box[pc].kind == pc_kind]

        types_sides = np.array([np.nan if s is None else (0 if s==0 else s.type) for s in sides])
        types_toFind = -np.tile(types_sides, 6)[:4*5].reshape(4, 5)[[0, 3, 2, 1], :4]
        types_random = np.isnan(types_toFind)

        potentials = []
        for pc in pcs_potential:
            test = np.logical_or(types_toFind == self.box[pc].types, types_random)
            found = np.where(test.all(axis=1))[0]
            if len(found) > 0:
                found_comb = list(itertools.product([pc], found))
                potentials.extend(found_comb)

        if len(potentials) > 0:
            to_test = np.where(np.isfinite(types_sides))[0]
            found_fits = pd.Series(index=pd.MultiIndex.from_arrays(zip(*potentials)))
            for pc, orient in found_fits.index:
                found_fits[(pc, orient)] = self.box[pc].fit(sides, orient=orient, to_test=to_test)
        else:
            found_fits = pd.Series()
        return found_fits

    def get_open_fields(self):
        return self.fields.applymap(lambda x: x[0] is None)

    def get_potential_positions(self):
        open_fld = self.get_open_fields()
        open_pos = np.array(zip(*np.where(open_fld)))
        filled_fld = np.logical_not(open_fld.values)
        test_fun = lambda t: filled_fld.flat[np.ravel_multi_index((t[:, 0], t[:, 1]), dims=(self.height, self.width))]
        if len(open_pos) > 0:
            neighbors = np.zeros(shape=(len(open_pos), 4), dtype=np.bool)
            ind_top = open_pos[:, 0] > 0
            neighbors[ind_top, 0] = test_fun(open_pos[ind_top, :] - [1, 0])
            ind_right = open_pos[:, 1] < self.width-1
            neighbors[ind_right, 1] = test_fun(open_pos[ind_right, :] + [0, 1])
            ind_bottom = open_pos[:, 0] < self.height-1
            neighbors[ind_bottom, 2] = test_fun(open_pos[ind_bottom, :] + [1, 0])
            ind_left = open_pos[:, 1] > 0
            neighbors[ind_left, 3] = test_fun(open_pos[ind_left, :] - [0, 1])
            pot_pos_array = open_pos[neighbors.any(axis=1), :]
            pot_pos = list(map(tuple, pot_pos_array))
        else:
            pot_pos = []
        return pot_pos

    def get_neighboring_sides(self, pos):
        actions = [(0, 0, (-1, 0), 2), (1, self.width-1, (0, 1), 3),
                   (0, self.height-1, (1, 0), 0), (1, 0, (0, -1), 1)]

        n = 4*[None]
        for i in range(4):
            t = actions[i]
            if pos[t[0]] != t[1]:
                pc_id, orient = self.fields.iloc[pos[0]+t[2][0], pos[1]+t[2][1]]
                if not (pc_id is None):
                    n[i] = self.box[pc_id].get_side(orient, t[3])
            else:
                n[i] = 0
        return n

    def try_to_fill_positions(self, pot_pos):
        fits = np.zeros(shape=len(pot_pos))
        for i in range(len(pot_pos)):
            fits[i] = self.try_to_fill_position(pot_pos[i])
        return fits

    def try_to_fill_position(self, pos, thres=config.fit_thres):
        found_fits = self.find_potential_fits(pos)
        if len(found_fits) > 0:
            winner = found_fits.argmax()
            winner_fit = found_fits.loc[winner]
            if winner_fit > thres:
                self.set_piece(winner[0], pos=pos, orient=winner[1], fit=winner_fit)
                return winner_fit
            else:
                return 0.
        else:
            return 0.

    def tree_find_fits(self, pos, depth, thres=config.fit_thres, pos_mask=None):
        """Returns:
        0 - found piece and set it to field
        1 - found piece, but fit was not good
        2 - no potential piece found
        """
        found_fits = self.find_potential_fits(pos)
        if len(found_fits) == 0:
            return 2
        count_sides = len([s for s in self.get_neighboring_sides(pos) if isinstance(s, pieces.Side)])
        potentials = found_fits.index[found_fits > thres**count_sides]
        if len(potentials) == 0:
            return 1
        tree_mask, count_sides = self.get_tree_mask(pos, depth)

        tree_fit = pd.Series(found_fits.loc[potentials], index=potentials)
        if depth > 0 and len(tree_mask) > 0:
            for pc, orient in potentials:
                brd = self.copy()
                brd.fit = 1.
                brd.set_piece(pc, pos=pos, orient=orient, fit=found_fits.loc[(pc, orient)])
                success = brd.solve_puzzle_single_fields(pos_mask=tree_mask)
                if success:
                    tree_fit.loc[(pc, orient)] = brd.fit
                else:
                    tree_fit.loc[(pc, orient)] = 0.
        winner = tree_fit.argmax()
        mean_fit = tree_fit.loc[winner]**(1./count_sides)
        if mean_fit >= thres:
            self.set_piece(winner[0], pos, winner[1], fit=found_fits.loc[winner])
            print('Filled field: pos=%s, id=%s, orient=%s, fit=%s, mean_fit=%s'\
                  %(pos, winner[0], winner[1], found_fits.loc[winner], mean_fit))
            return 0
        else:
            return 1

    def get_tree_mask(self, pos, depth, pos_mask=None):
        open = self.get_open_fields()
        open_sliced = open.loc[max(0, pos[0]-depth):min(self.height, pos[0]+depth),
                      max(0, pos[1]-depth):min(self.width, pos[1]+depth)]
        num_ind = np.where(open_sliced)
        tree_mask = zip(*(open.index[num_ind[0]], open.columns[num_ind[1]]))
        if not pos_mask is None:
            tree_mask = list(set(tree_mask).intersection(set(pos_mask)))

        tree_copy = deepcopy(tree_mask)
        count_sides = 0
        filled = np.array(np.where(np.logical_not(open))).T
        for pot in tree_copy:
            tree_array = np.array(tree_copy)
            count_sides += (np.abs(pot-tree_array).sum(axis=1) == 1).sum()
            count_sides += (np.abs(pot-filled).sum(axis=1) == 1).sum()
            tree_copy.remove(pot)
        return tree_mask, count_sides

    def show(self, save=True):
        matplotlib.rc('axes',edgecolor='w')

        for i in range(self.height):
            for j in range(self.width):
                ax = plt.subplot2grid((self.height, self.width), (i,j))
                ax.set_yticks([])
                ax.set_xticks([])
                pc, orient = self.fields.loc[i,j]
                if not pc is None:
                    self.box[pc].show(orient, ax=ax)
        plt.suptitle(self.name)
        plt.subplots_adjust(hspace=0, wspace=0)
        if save:
            plt.savefig(config.RESULTS_PATH + os.sep + self.name + '.png')
        else:
            plt.show()

    def try_to_solve_puzzle_tree(self, parallel=False, NIter=10000, depth=config.search_depth, pos_mask=None):
        brd = self.copy()
        done = False
        for iIter in range(NIter):
            pot_pos = brd.get_potential_positions()
            if not pos_mask is None:
                pot_pos = set(pot_pos).intersection(pos_mask)
            if len(pot_pos) == 0:
                done = True
                break
            if parallel:
                res = Parallel(n_jobs=4)(delayed(find_fits)(brd, pos, depth, pos_mask) for pos in pot_pos)
            else:
                res = []
                for pos in pot_pos:
                    res.append(find_fits(brd, pos, depth, pos_mask))
            states = zip(*res)[1]
            if not 0 in states:
                done = True
                break
        if done:
            self.fields = brd.fields
            self.determine_unused()
            return True
        else:
            print('Defeat!')
            return False