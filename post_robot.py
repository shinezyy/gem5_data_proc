#!/usr/bin/env python2.7

import os
import time
import re
import argparse

from extract import extract_stat
from mail import send
from paths import *

running_scrpits = [
    'select.sh',
    #'good.sh',
    #'gcc.sh',
]

content = []

def cat(x, y):
    return os.path.join(os.path.expanduser(x), y)


def extract_dirs():
    global content
    for script in running_scrpits:
        if not os.path.isfile(cat(script_dir(), script)):
            print script + ' is not a file in ' + script_dir()
            return None
        else:
            content = content + extract_script(cat(script_dir(), script))

    return content

def extract_script(filename):
    d = []
    with open(filename) as inf:
        for line in inf:
            p = re.compile('.*"(.*)\\\;(.*)" -o (.*) -s')
            m = p.match(line)
            pair = [m.group(1), m.group(2)]
            path = m.group(3)
            x = dict()
            x['pair'] = pair
            x['path'] = path
            d.append(x)
    return d


def main():
    parser = argparse.ArgumentParser(description='check sum of slots',
                                prefix_chars='-')

    parser.add_argument('-t', '--tail', action='store_true',
                        help='use statistics in tail of stats.txt')

    parser.add_argument('-b', '--brief', action='store_true',
                        help='display brief statistics')

    args = parser.parse_args()

    while True:
        s = ''
        for f in extract_dirs():
            s += str(f['pair']) + '\n'
            s += extract_stat(cat(f['path'], 'stats.txt'), args.tail,
                              cat(cat(st_stat_dir(), f['pair'][0]), 'stats.txt'),
                              args.brief)

        print s

        sender = 'diamondzyy@163.com'
        receiver = 'diamondzyy@sina.com'

        # send(sender, receiver, s)
        time.sleep(300)


if __name__ == '__main__':
    main()
