standard_targets = [
    '(numCycles)',
    '(rename\..+)_Slots',
    '(iew\..+)_Slots',
    'cpu\.(committedInsts::\d)',
    'fmt\.(num.*Slots::0)',
    '(cpu\.ipc::\d)',
    'cpu\.(HPTpredIPC)::0',
    '(cpu\.HPTQoS)',
    'fmt\.(mlp_rectification)',
    # 'iew.(branchMispredicts::0)',
    # 'iew.exec_(branches::0)',
    'rename\.(.+FullEvents::\d)',
    '(dcache.*_m)iss_rate::0',
    '(l2.*_m)iss_rate::0',
    '(iq_util)ization::[01]',
    '(rob_util)ization::[01]',
    'lsq\.(.+_util)ization::[01]',
]

brief_targets = [
    '(numCycles)',
    'cpu\.(committedInsts::0)',
    'fmt\.(num.*Slots::0)',
    'cpu\.(HPTpredIPC)::0',
    'fmt\.(mlp_rectification)',
]


