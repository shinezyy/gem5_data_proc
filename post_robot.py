#!/usr/bin/env python2.7

import os
import time
import re

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


while True:
    s = ''
    for f in extract_dirs():
        s += str(f['pair']) + '\n'
        s += extract_stat(cat(f['path'], 'stats.txt'), True,
                          cat(cat(st_stat_dir(), f['pair'][0]), 'stats.txt'))

    print s

    sender = 'diamondzyy@163.com'
    receiver = 'diamondzyy@sina.com'

    # send(sender, receiver, s)
    break
    time.sleep(300)

