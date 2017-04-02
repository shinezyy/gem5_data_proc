#!/usr/bin/env python2.7

import os
import time
import random as rd

batch = [
    'perlbench',
    'bzip2',
    'gcc',
    'mcf',
    'gobmk',
    'hmmer',
    'sjeng',
    'libquantum',
    'astar',
]

'''
sample = rd.sample(xrange(81), 24)

for s in sample:
    print batch[s / 9], batch[s % 9]
'''

for b in batch:
    print b, batch[0]
