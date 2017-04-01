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
    #'~/dyn_part_all2',

    # share tlb:
    #'~/dyn_share_tlb',
    #'~/dyn_share_tlb2',

    # share bp:
    #'~/dyn_share_bp',

    # front end control:
    #'~/fc_90',

    # combined control:
    '~/cc_90',
]

#file_name = './stat/pred_ipc_error_share_tlb.txt'
#file_name = './stat/pred_ipc_error_share_bp.txt'
#file_name = './stat/pred_ipc_error_part_all.txt'
#file_name = './stat/qos90_part_all_fc.txt'
#file_name = './stat/qos90_part_all_cc.txt'
#file_name = './stat/dyn_thp.txt'
#file_name = './stat/fc_thp.txt'
file_name = './stat/cc_thp.txt'

def gen_stat_path(p, hpt, lpt):
    return cat(cat(p, hpt+'_'+lpt), 'stats.txt')

result = []

for line in get_rand_list():
    qos = 0
    pred_qos = 0
    hpt, lpt = line
    for pd in possible_dirs:
        if os.path.isfile(gen_stat_path(pd, hpt, lpt)):
            try:
                hpt_ipc = specify_stat(gen_stat_path(pd, hpt, lpt),
                                       False, 'system.cpu.committedInsts::0')
                lpt_ipc = specify_stat(gen_stat_path(pd, hpt, lpt),
                                       False, 'system.cpu.committedInsts::1')
            except:
                print 'Unexpected error:', sys.exc_info()
                hpt_ipc = specify_stat(gen_stat_path(pd, hpt, lpt),
                                       True, 'system.cpu.committedInsts::0')
                lpt_ipc = specify_stat(gen_stat_path(pd, hpt, lpt),
                                       True, 'system.cpu.committedInsts::1')

            thp = float(hpt_ipc) + float(lpt_ipc)

    line.append(hpt_ipc)
    line.append(lpt_ipc)
    result.append(line)

# print 'avg:', np.mean(error_overall, axis=0), 'std:', np.std(error_overall, axis=0)
print result

with open(file_name, 'w') as f:
    for line in result:
        f.write(', '.join([x for x in line]) + '\n')
print 'saved from {} to {}'.format(possible_dirs[0], file_name)
