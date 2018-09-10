standard_targets = [
    '(numCycles)',
    # 'rename\.(.+FullEvents::\d)',
    'cpu\.(committedInsts::\d)',
    'fmt\.(num.*Slots::0)',
    'cpu\.(ipc::\d)',
    'cpu\.(HPTpredIPC)::0',
    '(cpu\.HPTQoS)',
    'fmt\.(mlp_rectification)',
]

slot_targets = [
    'cpu\.(committedInsts::\d)',
    '(fetch\..+)_Slots',
    '(decode\..+)_Slots',
    '(rename\..+)_Slots',
    #'(rename\..+W)aits::0',
    #'(rename\..+F)ullEvents::0',
    '(iew\..+)_Slots',
]

cache_targets = [
    '(dcache.*_m)iss_rate::0',
    '(icache.*_m)iss_rate::0',
    '(l2.*_m)iss_rate::0',
]

branch_targets = [
    'iew\.(branchMispredicts::0)',
    'iew\.exec_(branches::0)',
    'iew\.iewExec(LoadInsts::0)',
    'iew\.exec_(stores::0)',
    'thread(0\.squashedLoads)',
    'thread(0\.squashedStores)',
    '(iqSquashedInstsIssued)',
    '(commitSquashedInsts)',
]

util_targets = [
    '(iq_util)ization::0',
    '(rob_util)ization::0',
    'lsq\.(.+_util)ization::0',
]

brief_targets = [
    'switch_cpus\.(ipc)',
    'switch_cpus\.committed(Insts)',
    'switch_cpus\.iew\.IQFullUti(l::1)',
    'switch_cpus\.iew\.IQFullUti(l::2)',
    'switch_cpus\.iew\.IQFullUti(l::3)',
    'switch_cpus\.iq\.scoreBoard(Reads)',
    # 'switch_cpus\.iew\.IQFullReason::(TooMany)Operand',
    'switch_cpus\.iew\.IQFullReason::(P1C)onflict',
    'switch_cpus\.iew\.IQFullReason::(P2C)onflict',
    'switch_cpus\.iew\.IQFullReason::(P3C)onflict',
    'switch_cpus\.iew\.IQFullReason::(P1QP)ortLimit',
    'switch_cpus\.iew\.IQFullReason::(P2QP)ortLimit',
    'switch_cpus\.iew\.IQFullReason::(P3QP)ortLimit',
    # 'switch_cpus\.iew\.iewIQ(FullEvents)',
    'switch_cpus\.num(Cycles)'
]

ipc_target = [
    'switch_cpus\.(ipc)',
]


special_targets = [
    '(cpu\.HPTQoS)',
]
