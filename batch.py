#!/usr/bin/env python2.7

import os
import time
import re
import argparse

from paths import *
import common as c


def main():
    pairs = c.pairs('~/dyn_0425')
    pairs = c.stat_filt(pairs)
    pairs = c.time_filt(pairs)
    c.print_list(pairs)

if __name__ == '__main__':
    main()
