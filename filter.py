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
    #'~/dyn_bpp2',
    '~/dyn_64_lsq_special3',

    # share tlb:
    #'~/dyn_share_tlb',
    #'~/dyn_share_tlb2',

    # share bp:
    #'~/dyn_share_bp',
]

#file_name = './stat/pred_ipc_error_share_tlb.txt'
#file_name = './stat/pred_ipc_error_share_bp.txt'
#file_name = './stat/pred_ipc_error_part_all.txt'

def gen_stat_path(p, hpt, lpt):
    return cat(cat(p, hpt+'_'+lpt), 'stats.txt')

result = []

threshold = 0.9

for line in get_rand_list('./rand.txt'):
    hpt, lpt = line
    for pd in possible_dirs:
        if os.path.isfile(gen_stat_path(pd, hpt, lpt)):
            pred_qos = float(specify_stat(gen_stat_path(pd, hpt, lpt),
                                    False, 'system.cpu.HPTQoS'))

            smt_ipc = float(specify_stat(gen_stat_path(pd, hpt, lpt),
                                          False, 'cpu.ipc::0'))

            st_ipc = float(specify_stat(cat(cat(st_stat_dir(),
                                                hpt + '_perlbench'),
                                        'stats.txt'),
                                    False, 'system.cpu.HPTpredIPC::0'))
            real_qos = smt_ipc/st_ipc

            if real_qos < threshold:
                print hpt, lpt #, real_qos

