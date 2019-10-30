brief_targets = [
    'cpus\.(ipc)',
    'cpus\.committed(Insts)',
    #'cpus\.num(Cycles)'
]
ipc_target = [
    'cpus\.(ipc)',
]

flow_target = [
        'dataflow_queue\.(WKFlow)Usage::total'
        ]

standard_targets = [
    '(numCycles)',
    'cpus\.committed(Insts)',
    'cpus\.(ipc)',
]

cache_targets = [
    '(dcache.*_m)iss_rate',
    '(icache.*_m)iss_rate',
    '(l2.*_m)iss_rate',
]

branch_targets = [
    'd?iewc?\.?(branchMispredicts)',
    'd?iewc?\.?exec_(branches)',
    # 'iew\.iewExec(LoadInsts)',
    # 'iew\.exec_(stores)',
    # 'thread(0\.squashedLoads)',
    # 'thread(0\.squashedStores)',
    # '(iqSquashedInstsIssued)',
    # '(commitSquashedInsts)',
]
fanout_targets = [
        'diewc\.(largeFanoutInsts)',
        'diewc\.(falseNegativeLF)',
        'diewc\.(falsePositiveLF)',
        'diewc\.(forwarders)Committed',
        #'diewc\.(firstLevelFw)',
        #'diewc\.(secondaryLevelFw)',
        #'diewc\.(gainFromReshape)',
        'diewc\.(reshapeContrib)',
        'diewc\.(nonCriticalForward)',
        #'diewc\.(negativeContrib)',
        'diewc\.(wkDelayedCycles)',

        'cpus\.(squashedFUTime)',
        'dataflow_queue\.(readyWaitTime)::total',
        'dataflow_queue(oldWaitYoung)',
        ]
breakdown_targets= [
        'diewc\.(queueingD)elay',
        'diewc\.(ssrD)elay',
        'diewc\.(pendingD)elay',
        'diewc\.FU(ContentionD)elay',
        'dataflow_queue\.(HalfSquash)es',
        ]
