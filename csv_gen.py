#!/usr/bin/env python2.7

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


concerned = [
    'perlbench',
    'bzip2',
    'gcc',
    'hmmer',
    'sjeng',
]

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

short = {
    'perlbench' : 'perl',
    'libquantum' : 'libq',
}

possible_dirs = [
    '~/dyn_part_all',
    '~/dyn_bpp2',
]

matrix = dict()

def gen_stat_path(p, hpt, lpt):
    if hpt in short:
        hpt = short[hpt]
    if lpt in short:
        lpt = short[lpt]
    return cat(cat(p, hpt+'_'+lpt), 'stats.txt')

file_name = './stat/pred_ipc_error.csv'

with open(file_name, 'w') as f:
    # write headers
    header = 'concernedBenchmark, ' + ', '.join([lpt for lpt in batch]) + '\n'
    print header
    f.write(header)

    error_overall = 0

    for hpt in concerned:
        errors = []
        for lpt in batch:
            for pd in possible_dirs:
                if os.path.isfile(gen_stat_path(pd, hpt, lpt)):
                    error = 10000
                    try:
                        pred_ipc = specify_stat(gen_stat_path(pd, hpt, lpt),
                                                False, 'system.cpu.HPTpredIPC::0')
                    except:
                        print 'Unexpected error:', sys.exc_info()
                        pred_ipc = specify_stat(gen_stat_path(pd, hpt, lpt),
                                                True, 'system.cpu.HPTpredIPC::0')

                    real_ipc = specify_stat(cat(cat(st_stat_dir(), hpt),
                                                'stats.txt'),
                                            False, 'system.cpu.HPTpredIPC::0')

                    error = abs(float(pred_ipc) - float(real_ipc))/float(real_ipc)
                    error_overall += error

                    errors.append(error)
        f.write(hpt + ', ' + ', '.join([str(error) for error in errors]) + '\n')
    print error_overall/ (len(concerned)*len(batch))

