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

for b in [1, 6, 7]:
    sample = rd.sample(xrange(9), 3)
    print batch[b], batch[b]
    for s in sample:
        if s!= b:
            print batch[b], batch[s]
