#!/usr/bin/env python2.7

import os
import time
import random as rd

batch = [
    'libquantum',
    'gobmk',
    'sjeng',
    'bzip2',
    'perlbench',
    'gcc',
    'mcf',
    'hmmer',
    'astar',
]

'''
sample = rd.sample(xrange(81), 24)

for s in sample:
    print batch[s / 9], batch[s % 9]
'''

'''
for b in batch:
    print b, batch[0]
'''
for x in batch[0: 4]:
    for y in batch[0: 4]:
        print x, y
