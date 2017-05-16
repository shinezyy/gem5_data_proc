standard_targets = [
    '(numCycles)',
    'rename\.(.+FullEvents::0)',
    'cpu\.(committedInsts::\d)',
    'fmt\.(num.*Slots::0)',
    'cpu\.(ipc::\d)',
    'cpu\.(HPTpredIPC)::0',
    '(cpu\.HPTQoS)',
    'fmt\.(mlp_rectification)',
]

slot_targets = [
    'cpu\.(committedInsts::\d)',
    # '(fetch\..+)_Slots',
    '(decode\..+)_Slots',
    '(rename\..+)_Slots',
    '(rename\..+W)aits::0',
    '(rename\..+F)ullEvents::0',
    '(iew\..+)_Slots',
]

cache_targets = [
    '(dcache.*_m)iss_rate::0',
    '(icache.*_m)iss_rate::0',
    '(l2.*_m)iss_rate::0',
]

branch_targets = [
    'iew.(branchMispredicts::0)',
    'iew.exec_(branches::0)',
]

util_targets = [
    '(iq_util)ization::0',
    '(rob_util)ization::0',
    'lsq\.(.+_util)ization::0',
]

brief_targets = [
    '(numCycles)',
    'cpu\.(committedInsts::0)',
    'fmt\.(num.*Slots::0)',
    'cpu\.(HPTpredIPC)::0',
    'fmt\.(mlp_rectification)',
    'cpu\.(ipc::0)',
]


