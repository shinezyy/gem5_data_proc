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
    'rename\.(.+FullEvents::\d)',
    '(dcache.*_miss_rate::0)',
    '(l2.*_miss_rate::0)',
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
        ['cpu.committedInsts::0', 'hpt_insts'],
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
        '''
        raw_str = sh.grep("system.cpu.committedInsts::0 *" +\
                          str(num_insts) + "[0-9]\{6\}",
                          stat_file, '-m', "1", '-A', '1000', '-B', '600')
        '''

        for x in range(0, 400):
            x = int(num_insts) + (x/2) * ((-1) ** x)
            try:
                t = " *{}".format(x) + "[0-9]\{" + str(5) + "\} "
                print t
                first = sh.grep("system.cpu.committedInsts::0" + t,
                                stat_file, '-m', "1", '-A',
                                '1000', '-B', '600')
                raw_str = first
                break
            except:
                continue



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

        named = named_stat()
        for k in named:
            d[named[k]] = float(d[k])

        # Get ST IPC from speculated file

        dyn_insts_now = int(d['hpt_insts'])
        # print 'getting around num_insts', dyn_insts_now
        num_digits = len(str(dyn_insts_now))
        msd = dyn_insts_now / 10 ** (num_digits-3)

        for x in range(0, 400):
            x = msd + (x/2) * ((-1) ** x)
            try:
                t = " *{}".format(x) + "[0-9]\{" + str(num_digits-3) + "\} "
                print t
                first = sh.grep("system.cpu.committedInsts::0" + t,
                    st_stat_file, '-m', "1", '-A', '1000', '-B', '600')
            except:
                continue
            # print first
            if first:
                # print 'hit at *{}'.format(x) + "[0-9]\{5\}"
                # st_raw_str = sh.grep(first, 'ipc::0')
                st_raw_str = str(first)
                break

        '''
        st_raw_str = sh.tail(st_stat_file, '-n', '2000')
        '''

        for line in st_raw_str.split('\n'):
            m = re.match('(system.cpu.ipc::0) +(\d+\.?\d*)', line)
            if not m is None:
                d['st_ipc'] = float(m.group(2))

            x = re.match('system\.(cpu.committedInsts::0) + (\d*)', line)
            if not x is None:
                # print 'Matched'
                d['st_insts'] = float(x.group(2))

        ret += '----st.ipc (around instructions {})'. \
                format(d['st_insts']).ljust(30) \
                    + str(d['st_ipc']) + '\n'

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

