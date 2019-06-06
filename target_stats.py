brief_targets = [
    'cpus\.(ipc)',
    'cpus\.committed(Insts)',
    'cpus\.num(Cycles)'
]


standard_targets = [
    '(numCycles)',
    'cpus\.(committedInsts)',
    'cpus\.(ipc)',
]

cache_targets = [
    '(dcache.*_m)iss_rate',
    '(icache.*_m)iss_rate',
    '(l2.*_m)iss_rate',
]

branch_targets = [
    'iew\.(branchMispredicts)',
    'iew\.exec_(branches)',
    # 'iew\.iewExec(LoadInsts)',
    # 'iew\.exec_(stores)',
    # 'thread(0\.squashedLoads)',
    # 'thread(0\.squashedStores)',
    # '(iqSquashedInstsIssued)',
    # '(commitSquashedInsts)',
]
