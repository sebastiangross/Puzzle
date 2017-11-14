__author__ = 'Herakles II'

from source import board
import time

#ToDo 2: Corner Finder
#ToDo 4: Variable Search depth
#ToDo 5: Logging

def solve(name):
    brd = board.Board(name=name)
    brd.load_pieces()

    brd.obtain_border()
    brd.try_to_solve_puzzle_tree(parallel=False, depth=1)

if __name__ == '__main__':
    parallel = False

    #import cProfile
    #command = """solve('DSC01132_10x15')"""
    #cProfile.runctx(command, globals(), locals(), filename="'DSC01132_10x15_3.profile" )

    #name = 'Bild_3_5x10'
    #name = 'DSC01132_10x15'
    #name = 'DSC01132_25x40'
    #name = 'DSC01132_15x30'
    #name = 'DSC01132_15x30_b'
    #name = 'DSC01132_15x30_nb'
    #name = 'DSC01132_5x10'
    name = 'DSC01132_3x5'

    brd = board.Board(name=name)
    brd.load_pieces()
    #brd.test_all_sides(parallel=False)

    ##normal
    obtained_border = brd.obtain_border()

    ##b
    #import itertools
    #for i, j in itertools.product([0, brd.height-1], range(brd.width)):
    #    brd.set_piece('p_%s_%s'%(i, j), pos=(i, j), orient=0)
    #for i, j in itertools.product(range(1, brd.height-1), [0, brd.width-1]):
    #    brd.set_piece('p_%s_%s'%(i, j), pos=(i, j), orient=0)

    brd.try_to_solve_puzzle_tree(parallel=False, depth=1)

    brd.fields.to_csv('results\\' + name + '.csv')
    brd.show(save=False)