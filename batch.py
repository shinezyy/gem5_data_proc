#!/usr/bin/env python3.6

import os
from os.path import join as pjoin
import time
import re
import argparse

from paths import *
import common as c
from target_stats import *


def main():
    paths = c.pairs('~/dyn_0425')
    paths = c.stat_filt(paths)
    paths = c.time_filt(paths)
    paths = [pjoin(x, 'stats.txt') for x in paths]
    pairs = c.pairs('~/dyn_0425', return_path=False)

    for pair, path in zip(pairs, paths):
        d = c.get_stats(path, brief_targets, re_targets=True)
        print(d)

if __name__ == '__main__':
    main()
