#!/usr/bin/env python2.7

import os
import time
import re
import argparse
import sys
import numpy as np

from extract import extract_stat
from specify import specify_stat
from mail import send
from paths import *

def cat(x, y):
    return os.path.join(os.path.expanduser(x), y)


possible_dirs = [
    # part all:
    #'~/dyn_part_all2',
    #'~/dyn_bpp2',
    '~/dyn_64_lsq_hard',

    # share tlb:
    #'~/dyn_share_tlb2',

    # controlled:
    #'~/sim_st_64_lsq',
]

pairs = './hard.txt'

short = {
    'perlbench' : 'perl',
    'libquantum' : 'libq',
}

matrix = dict()

def gen_stat_path(p, hpt, lpt):
    return cat(cat(p, hpt+'_'+lpt), 'stats.txt')

def get_rand_list(p):
    ret = []
    with open(p) as f:
        for line in f:
            ret.append(line.split())
    return ret

if __name__ == '__main__':
    count = 0
    for hpt, lpt in get_rand_list(pairs):
        ready = False
        for pd in possible_dirs:
            if os.path.isfile(gen_stat_path(pd, hpt, lpt)):
                try:
                    pred_ipc = specify_stat(gen_stat_path(pd, hpt, lpt),
                                            False, 'system.cpu.HPTpredIPC::0')
                    ready = True
                    print hpt, lpt, 'is ready in', pd
                    count += 1
                except:
                    ready = ready
                    print hpt, lpt, 'Not Ready'
    print count, 'ready'
