#!/usr/bin/env python3.6

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
    #'~/dyn_bpp2',

    # share tlb:
    #'~/dyn_share_tlb',
    #'~/dyn_share_tlb2',

    # share bp:
    #'~/dyn_share_bp',

    # front end control:
    #'~/fc_70',

    # combined control:
    #'~/hard_cc_90',
    #'~/qos80_cc_64_lsq',
    #'~/qos90_fc_64_lsq',
    #'~/timely_update_cc_64_lsq/cc_64_lsq'
    #'~/old_results_2017_0403/cc_90'
    '~/cc_80'
]

#file_name = './stat/pred_ipc_error_share_tlb.txt'
#file_name = './stat/pred_ipc_error_share_bp.txt'
#file_name = './stat/pred_ipc_error_part_all.txt'
#file_name = './stat/qos70_part_all_fc.txt'
#file_name = './stat/qos90_64_fc.txt'
file_name = './stat/qos80_cc.txt'
#file_name = './stat/qos90_fc.txt'
#file_name = './stat/qos80_part_all_cc.txt'

def gen_stat_path(p, hpt, lpt):
    return cat(cat(p, hpt+'_'+lpt), 'stats.txt')

result = []

for line in get_rand_list('./rand.txt'):
    qos = 0
    pred_qos = 0
    hpt, lpt = line
    for pd in possible_dirs:
        if os.path.isfile(gen_stat_path(pd, hpt, lpt)):
            try:
                smt_ipc = specify_stat(gen_stat_path(pd, hpt, lpt),
                                        False, 'system.cpu.ipc::0')
                pred_qos = specify_stat(gen_stat_path(pd, hpt, lpt),
                                        False, 'system.cpu.HPTQoS')
            except:
                print('Unexpected error:', sys.exc_info())
                smt_ipc = specify_stat(gen_stat_path(pd, hpt, lpt),
                                        True, 'system.cpu.ipc::0')
                pred_qos = specify_stat(gen_stat_path(pd, hpt, lpt),
                                        True, 'system.cpu.HPTQoS')

            st_ipc = specify_stat(cat(cat(st_stat_dir(),
                                          hpt),
                                          #hpt + '_perlbench'),
                                      'stats.txt'),
                                  False, 'system.cpu.ipc::0')
            if smt_ipc and st_ipc:
                qos = float(smt_ipc)/float(st_ipc)
            else:
                qos = 10000

    if (pred_qos):
        line.append(str(qos))
        line.append(pred_qos)
        result.append(line)

# print 'avg:', np.mean(error_overall, axis=0), 'std:', np.std(error_overall, axis=0)
print(result)

with open(file_name, 'w') as f:
    for line in result:
        f.write(', '.join([x for x in line]) + '\n')
print('saved from {} to {}'.format(possible_dirs[0], file_name))
