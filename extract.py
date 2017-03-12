import sh



def shorter(x):
    ret = ''
    for line in str(x).split('\n'):
        first_half = line.split('#')[0].rstrip(' ')
        ret = ret + first_half + '\n'
    return ret


def extract_stat(stat_file, use_tail, targets, st_stat_file):

    ret = ''

    tstr = 'QoS'
    for s in targets:
        tstr += '\|'+s

    if use_tail == True:
        raw_str = sh.grep(sh.tail(stat_file, '-n', '2000'), tstr)
    else:
        raw_str = sh.grep(sh.grep("system.cpu.committedInsts::0 *2[0-9]\{8\}",
                        stat_file, '-m', "1", '-A', '1000', '-B', '600'), tstr)


    st_raw_str= sh.grep(sh.grep("system.cpu.committedInsts::0 *2[0-9]\{8\}",
                              st_stat_file, '-m', "1", '-A', '1000', '-B', '600'),
                        'ipc::')

    ret += shorter(raw_str)
    ret += shorter(st_raw_str)

    d = dict()

    for line in raw_str:
        line = str(line)
        if line.startswith('system.cpu.numCycles'):
           d['cycle'] = float(line.split()[1])
        elif line.startswith('system.cpu.fmt.numBaseSlots::0'):
            d['base'] = float(line.split()[1])
        elif line.startswith('system.cpu.fmt.numWaitSlots::0'):
            d['wait'] = float(line.split()[1])
        elif line.startswith('system.cpu.fmt.numMissSlots::0'):
            d['miss'] = float(line.split()[1])
        elif line.startswith('system.cpu.HPTpredIPC::0'):
            d['pred_ipc'] = float(line.split()[1])

    ret += 'Slot sanity: ' + \
            str((d['base'] + d['wait'] + d['miss'])/ (8*d['cycle'])) + '\n'

    for line in st_raw_str:
        line = str(line)
        if line.startswith('system.cpu.ipc::0'):
            d['st_ipc'] = float(line.split()[1])


    ret += 'Prediction error: ' + \
            str((d['pred_ipc'] - d['st_ipc']) / d['st_ipc']) + '\n'

    ret += '\n==========================================================\n'

    return ret

'''

for line in x:
    line = str(line)
    if line.startswith('system.cpu.numCycles'):
        d['system.cpu.numCycles'] = line.split()[1]
    elif line.startswith('system.cpu.fmt.numBaseSlots::0'):
        d['system.cpu.fmt.numBaseSlots::0'] = line.split()[1]
    elif line.startswith('system.cpu.fmt.numWaitSlots::0'):
        d['system.cpu.fmt.numWaitSlots::0'] = line.split()[1]
    elif line.startswith('system.cpu.fmt.numMissSlots::0'):
        d['system.cpu.fmt.numMissSlots::0'] = line.split()[1]
    elif line.startswith('system.cpu.ipc::0'):
        d['system.cpu.ipc::0'] = line.split()[1]
    elif line.startswith('system.cpu.iew.exec_branches::0'):
        d['hptBranch'] = line.split()[1]
    elif line.startswith('system.cpu.iew.branchMispredicts::0'):
        d['hptMiss'] = line.split()[1]
    elif line.startswith('system.cpu.itb.fetch_accesses::0'):
        itb_all = float(line.split()[1])
    elif line.startswith('system.cpu.itb.fetch_misses::0'):
        itb_miss = float(line.split()[1])

base = float(d['system.cpu.fmt.numBaseSlots::0'])
wait = float(d['system.cpu.fmt.numWaitSlots::0'])
miss = float(d['system.cpu.fmt.numMissSlots::0'])
cycle = float(d['system.cpu.numCycles'])
ipc = float(d['system.cpu.ipc::0'])

print (base + wait + miss)/ (8*cycle)

qos = (base + miss) / (base + miss + wait)

print "Predicted QoS:", qos

print 'Predicted ST IPC:', ipc/qos

'''
