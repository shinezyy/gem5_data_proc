import os
import time

from extract import extract_stat
from mail import send
from targets import targets

script_dir = '/home/zhouyaoyang/projects/dev_pard_smt/smt_run'

st_stat_dir = '/home/zhouyaoyang/sim_st_0304'

running_scrpits = [
    'select.sh',
    #'good.sh',
    #'gcc.sh',
]

targets = [
    'committedInsts::',
    'Slots::0',
    # 'cpi::',
    'ipc::',
    'numCycles',
    'system.cpu.rename.*_Slots',
    'system.cpu.HPTpredIPC::0',
    'HPTQoS',
    # 'system.cpu.*_utilization',
]

content = []

def cat(x, y):
    return os.path.join(os.path.expanduser(x), y)


def extract_dirs():
    global content
    for script in running_scrpits:
        if not os.path.isfile(cat(script_dir, script)):
            print script + ' is not a file in ' + script_dir
            return None
        else:
            content = content + extract_script(cat(script_dir, script))

    return content

def extract_script(filename):
    d = []
    with open(filename) as inf:
        for line in inf:
            line = line.split('-b "')[1]
            pair, path = line.split('"')
            pair = pair.split('\\;')
            path = path.split('-o')[1].split('-s')[0].strip(' ')
            x = dict()
            x['pair'] = pair
            x['path'] = path
            d.append(x)
    return d


while True:
    s = ''
    for f in extract_dirs():
        s += str(f['pair']) + '\n'
        s += extract_stat(cat(f['path'], 'stats.txt'), True, targets,
                          cat(cat(st_stat_dir, f['pair'][0]), 'stats.txt'))

    print s

    sender = 'diamondzyy@163.com'
    receiver = 'diamondzyy@sina.com'

    # send(sender, receiver, s)
    break
    time.sleep(300)

