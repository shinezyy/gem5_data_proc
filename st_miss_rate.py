#!/usr/bin/env python3.6

import os
import time
import re
import argparse
import sys

from extract import extract_stat
from specify import specify_stat
from mail import send
from paths import *

def cat(x, y):
    return os.path.join(os.path.expanduser(x), y)


benchmarks = [
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

short = {
    'perlbench' : 'perl',
    'libquantum' : 'libq',
}

matrix = dict()

def gen_stat_path(p, b):
    return cat(cat(p, b), 'stats.txt')

file_name = './stat/st_miss_rate.csv'

targets = {
    'L2 Cache Miss Rate' : 'system.l2.overall_miss_rate::0',
    'L1 D-Cache Miss Rate' : 'system.cpu.dcache.overall_miss_rate::0',
    'L1 I-Cache Miss Rate' : 'system.cpu.icache.overall_miss_rate::0',
}


with open(file_name, 'w') as f:
    header = ' ' + ', ' + \
            ', '.join([benchmark for benchmark in benchmarks]) + '\n'
    print(header)
    f.write(header)

    for k in targets:
        # write headers

        miss_rates = []
        for b in benchmarks:
            miss_rate = specify_stat(gen_stat_path(st_stat_dir, b),
                                    False, targets[k])
            miss_rates.append(miss_rate)

        f.write(k + ', ' + ', '.join(miss_rates) + '\n')
