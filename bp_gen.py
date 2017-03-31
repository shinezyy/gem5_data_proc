#!/usr/bin/env python2.7

import os
import time
import re
import argparse
import sys
import numpy as np

from extract import extract_stat
from specify import specify_stat
from check_ready import get_rand_list
from paths import *

def cat(x, y):
    return os.path.join(os.path.expanduser(x), y)


possible_dirs = [
    # part all:
    '~/dyn_part_all2',
    #'~/dyn_bpp2',

    # share tlb:
    #'~/dyn_share_tlb',
    #'~/dyn_share_tlb2',

    # share bp:
    #'~/dyn_share_bp',
]

file_name = './stat/mis_pred_rate_part_all.txt'

def gen_stat_path(p, hpt, lpt):
    return cat(cat(p, hpt+'_'+lpt), 'stats.txt')

result = []

for line in get_rand_list():
    error = 10000
    hpt, lpt = line
    for pd in possible_dirs:
        if os.path.isfile(gen_stat_path(pd, hpt, lpt)):
            miss_pred = specify_stat(gen_stat_path(pd, hpt, lpt),
                                     False, 'system.cpu.iew.branchMispredicts::0')
            all_branch = specify_stat(gen_stat_path(pd, hpt, lpt),
                                      False, 'system.cpu.iew.exec_branches::0')
            error = float(miss_pred)/float(all_branch)
    line.append(str(error))
    result.append(line)

print result

with open(file_name, 'w') as f:
    for line in result:
        f.write(', '.join([x for x in line]) + '\n')
