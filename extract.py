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
    '(iew\..+Slots)',
    '(cpu\.committedInsts::\d)',
    'fmt\.(num.*Slots::0)',
    '(cpu\.ipc::\d)',
    '(cpu\.HPTpredIPC::0)',
    '(cpu\.HPTQoS)',
    'fmt\.(mlp_rectification)',
    'iew.(branchMispredicts::0)',
    'iew.exec_(branches::0)',
    'rename\.(.+FullEvents::0)',
    # 'cpu\.(.+_utilization::[01])',
]

brief_patterns = [
    '(numCycles)',
    '(cpu\.committedInsts::\d)',
    'fmt\.(num.*Slots::0)',
    '(cpu\.HPTpredIPC::0)',
    'fmt\.(mlp_rectification)',
]


def preproc_pattern(p):
    p += ' +(\d+.?\d*)'
    return re.compile(p)


def named_stat():
    named_list = [
        ['numCycles', 'cycle'],
        ['numBaseSlots::0', 'base'],
        ['numWaitSlots::0', 'wait'],
        ['numMissSlots::0', 'miss'],
        ['cpu.HPTpredIPC::0', 'pred_ipc'],
        ['branchMispredicts::0', 'numBranchMiss'],
        ['branches::0', 'numBranch'],
    ]
    stat = dict()

    for p in named_list:
        stat[p[0].ljust(30)] = p[1]

    return stat


def extract_stat(stat_file, use_tail, st_stat_file, num_insts=0, brief = False):

    global patterns
    global brief_patterns

    if brief:
        compiled = map(preproc_pattern, brief_patterns)
    else:
        compiled = map(preproc_pattern, patterns)


    ret = ''

    # Get lines @ specified number of instructions
    if use_tail:
        raw_str = sh.tail(stat_file, '-n', '2000')
    elif num_insts == 0:
        raw_str = sh.grep("system.cpu.committedInsts::0 *2[0-9]\{8\}",
                        stat_file, '-m', "1", '-A', '1000', '-B', '600')
    else:
        raw_str = sh.grep("system.cpu.committedInsts::0 *" + str(num_insts) +
                          "[0-9]\{7\}",
                        stat_file, '-m', "1", '-A', '1000', '-B', '600')

    d = dict()

    # convert lines to dict for further use
    for line in raw_str:
        line = str(line)
        for p in compiled:
            m = p.search(line)
            if not m is None:
                d[m.group(1).ljust(30)] = m.group(2).rjust(20)
                ret += m.group(1).ljust(30) + m.group(2).rjust(20) + '\n'


    if not brief:
        # Get ST IPC from speculated file
        '''
        st_raw_str = sh.grep(sh.grep("system.cpu.committedInsts::0 *2[0-9]\{8\}",
                                    st_stat_file, '-m', "1", '-A', '1000', '-B', '600'),
                            'ipc::0')
        '''
        st_raw_str = sh.tail(st_stat_file, '-n', '2000')

        for line in st_raw_str:
            line = str(line)
            m = re.match('(system.cpu.ipc::0) +(\d+\.?\d*)', line)
            if not m is None:
                d['st_ipc'] = float(m.group(2))
                ret += '----st.ipc'.ljust(30) + m.group(2).rjust(20) + '\n'

        # check slot sanity
        named = named_stat()
        for k in named:
            d[named[k]] = float(d[k])

        ret += '\nSlot sanity: ' + \
                str((d['base'] + d['wait'] + d['miss'])/ (8*d['cycle'])) + '\n'

        # compute prediction error
        ret += 'Prediction error: ' + \
                str((d['pred_ipc'] - d['st_ipc']) / d['st_ipc']) + '\n'

        # compute branch misprediction rate
        ret += 'Branch misprediction rate: ' + \
                str(d['numBranchMiss'] / d['numBranch']) + '\n'

    ret += '\n==========================================================\n'

    return ret

