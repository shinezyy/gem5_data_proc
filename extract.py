import sh
import re


def shorter(x):
    ret = ''
    for line in str(x).split('\n'):
        first_half = line.split('#')[0].rstrip(' ')
        ret = ret + first_half + '\n'
    return ret


patterns = [
    '(numCycles)',
    '(rename\..+Slots)',
    '(cpu\.committedInsts::\d)',
    '(fmt\.num.*Slots::0)',
    '(cpu\.ipc::\d)',
    '(cpu\.HPTpredIPC::0)',
    '(cpu\.HPTQoS)',
]


def preproc_pattern(p):
    p += ' +(\d+.?\d*)'
    return re.compile(p)


def named_stat():
    named_list = [
        ['numCycles', 'cycle'],
        ['fmt.numBaseSlots::0', 'base'],
        ['fmt.numWaitSlots::0', 'wait'],
        ['fmt.numMissSlots::0', 'miss'],
        ['cpu.HPTpredIPC::0', 'pred_ipc'],
    ]
    stat = dict()

    for p in named_list:
        stat[p[0].ljust(30)] = p[1]

    return stat


def extract_stat(stat_file, use_tail, st_stat_file):

    global patterns

    compiled = map(preproc_pattern, patterns)

    ret = ''

    if use_tail == True:
        raw_str = sh.tail(stat_file, '-n', '2000')
    else:
        raw_str = sh.grep("system.cpu.committedInsts::0 *2[0-9]\{8\}",
                        stat_file, '-m', "1", '-A', '1000', '-B', '600')


    st_raw_str= sh.grep(sh.grep("system.cpu.committedInsts::0 *2[0-9]\{8\}",
                              st_stat_file, '-m', "1", '-A', '1000', '-B', '600'),
                        'ipc::0')

    d = dict()

    for line in raw_str:
        line = str(line)
        for p in compiled:
            m = p.search(line)
            if not m is None:
                d[m.group(1).ljust(30)] = m.group(2).rjust(20)
                ret += m.group(1).ljust(30) + m.group(2).rjust(20) + '\n'

    named = named_stat()
    for k in named:
        d[named[k]] = float(d[k])

    for line in st_raw_str:
        line = str(line)
        m = re.match('(system.cpu.ipc::0) +(\d+\.?\d*)', line)
        if not m is None:
            d['st_ipc'] = float(m.group(2))
            ret += '----st.ipc'.ljust(30) + m.group(2).rjust(20) + '\n'


    ret += '\nSlot sanity: ' + \
            str((d['base'] + d['wait'] + d['miss'])/ (8*d['cycle'])) + '\n'

    ret += 'Prediction error: ' + \
            str((d['pred_ipc'] - d['st_ipc']) / d['st_ipc']) + '\n'

    ret += '\n==========================================================\n'

    return ret

