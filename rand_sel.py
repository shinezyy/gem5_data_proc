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
    'specrand',
]

sample = rd.sample(xrange(100), 24)

for s in sample:
    print batch[s / 10], batch[s % 10]
