brief_targets = [
    '(?:cpus?|switch_cpus_1)\.(ipc)',
    # '(?:cpus?|switch_cpus_1)\.(cpi)',
    '(?:cpus?|switch_cpus_1)\.committed(Insts)',
    # '(?:cpus?|switch_cpus_1)\.(lastCommitTick)',
    # 'host_(inst_rate)',
    #'cpus\.num(Cycles)'
]
ipc_target = [
    '(?:cpus?|switch_cpus_1)\.(ipc)',
    '(?:cpus?|switch_cpus_1)\.(cpi)',
]

flow_target = [
    'DQGroup\d\.(WKFlowUsage::\d+)',
    'DQGroup\d\.(WKFlowUsage::\w+)',
        ]

standard_targets = [
    '(numCycles)',
    '(?:cpus?|switch_cpus_1)?\.committed(Insts)',
    '(?:cpus?|switch_cpus_1)?\.(ipc)',
]

icache_targets = [
        '(icache\.demandMisses)::total',
        '(icache\.overallAccesses)::total',
        ]

cache_targets = [
    # '(l3\.demandMissRate)::total',
    '(l3\.demandAccesses::l2\.pref)etcher',
    '(l3\.demandAcc)esses::total',
    '(l3\.demandMisses::l2\.pref)etcher',
    '(l3\.demandMiss)es::total',
    # '(l3\.overallAcc)esses::total',
    # '(l3\.overallMis)ses::total',
    # '(l2\.demandMissRate)::total',
    '(l2\.demandMiss)es::total',
    '(l2\.overallMiss)es::total',
    '(l2\.overallAcc)esses::total',
    '(l2\.demandAcc)esses::total',
    # 'cpu\.(dcache\.demandAvgMissLatency)::\.cpu\.data',
    # 'cpu\.(dcache\.demandMisses)::\.cpu\.data',
    # 'cpu\.(dcache.demandMissRate)::\.cpu\.data',
    # 'cpu\.(dcache\.overallAcc)esses::cpu\.data',
    # 'cpu\.(dcache\.demandAcc)esses::total',
    'cpu\.(dcache\.demandMiss)es::total',
    # 'cpu\.iew\.iew(ExecLoadInsts)',
]

topdown_targets = [
]

def add_topdown_targets():
    stalls = [
        'Nostall',
        'IcacheStall',
        'ITlbStall',
        'DTlbStall',
        'BpStall',
        'IntStall',
        'TrapStall',
        'FragStall',
        'SquashStall',
        'FetchBufferInvalid',
        'InstMisPred',
        'InstSquashed',
        'SerializeStall',
        'LongExecute',
        'InstNotReady',
        'LoadL1Stall',
        'LoadL2Stall',
        'LoadL3Stall',
        'StoreL1Stall',
        'StoreL2Stall',
        'StoreL3Stall',
        'ResumeUnblock',
        'CommitSquash',
    ]
    for stall in stalls:
        topdown_targets.append(r'system\.cpu\.iew\.dispatchStallReason::({})'.format(stall))

add_topdown_targets()


warmup_targets = [
    '(?:cpus?|switch_cpus_1)\.(?:diewc|commit)\.(branchMispredicts)',
    '(?:cpus?|switch_cpus_1)\.(?:diewc|commit)\.(branches)',
    '(l3\.demandMisses)::total',
    '(l2\.demandMisses)::total',
    '(numCycles)',
    '(?:cpus?|switch_cpus_1)?\.committed(Insts)',
]

branch_targets = [
    '(?:cpus?|switch_cpus_1)\.(?:diewxc|commit|iewx)\.(branchMispredicts)',
    '(?:cpus?|switch_cpus_1)?\.(?:diewxc\.exec_|commit\.)(branches)',
    '(?:cpus?|switch_cpus_1)?\.branchPred\.(indirectMispred)icted',
    '(?:cpus?|switch_cpus_1)?\.branchPred\.(RASIncorrect)',
    # 'cpu\.commit\.(branches)',
    # 'cpu\.commit\.(branchMispredicts)',
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

fetch_targets = [
    'cpus?\.fetch\.(fetchFromLoopBuffer)',
    'cpus?\.(fetch\.rate) ',
]


operand_targets = [
        # 'arch_state\.(numBusyOperands::\d)',
        'arch_state\.(numDispInsts::\d)',
        'arch_state\.(meanBusyOp_\d)',

        ]
packet_targets = [
        'DQGroup0\.(KeySrcP)acket',
        'DQGroup0\.(SrcOpP)ackets',
        'DQGroup0\.(DestOpP)ackets',
        'DQGroup0\.(MemP)ackets',
        'DQGroup0\.(OrderP)ackets',
        'DQGroup0\.(MiscP)ackets',
        'DQGroup0\.(TotalP)ackets',
        ]
breakdown_targets= [
        'diewc\.(queueingDelay)',
        'diewc\.(ssrDelay)',
        'diewc\.(pendingDelay)',
        # 'diewc\.FU(ContentionD)elay',
        'diewc\.(readyExecDelayTicks)',
        # 'dataflow_queue\.(HalfSquash)es',
        ]

model_targets = [
    r'cpus\.(ipc)',
    r'DQGroup(\d\.TotalPackets)',
    r'cpus\.committed(Insts)',
]

ff_power_targets = [

    r'arch_state\.(CombRename)',
    r'arch_state\.(RegReadCommitSB)',
    r'arch_state\.(RegWriteCommitSB)',
    r'arch_state\.(RegReadSpecSB)',
    r'arch_state\.(RegWriteSpecSB)',
    r'arch_state\.(RegReadMap)',
    r'arch_state\.(RegWriteMap)',
    r'arch_state\.(RegReadRT)',
    r'arch_state\.(RegWriteRT)',
    r'arch_state\.(RegReadSpecRT)',
    r'arch_state\.(RegWriteSpecRT)',
    r'arch_state\.(SRAMWriteMap)',
    r'arch_state\.(SRAMWriteSB)',
    r'arch_state\.(SRAMWriteRT)',
    r'arch_state\.(SRAMReadMap)',
    r'arch_state\.(SRAMReadSB)',
    r'arch_state\.(SRAMReadRT)',
    r'arch_state\.(RegReadARF)',
    r'arch_state\.(RegWriteARF)',
    r'arch_state\.(RegReadSpecARF)',
    r'arch_state\.(RegWriteSpecARF)',
    r'DQTop\.(RegReadCenterInstBuf)',
    r'DQTop\.(RegReadCenterPairBuf)',
    r'DQTop\.(RegReadCenterWKBuf)',
    r'DQTop\.(RegReadInterGroupWKBuf)',
    r'DQTop\.(RegWriteCenterInstBuf)',
    r'DQTop\.(RegWriteCenterPairBuf)',
    r'DQTop\.(RegWriteCenterWKBuf)',
    r'DQTop\.(RegWriteInterGroupWKBuf)',
    r'DQGrou(p\d\.QueueWriteTxBuf)',
    r'DQGrou(p\d\.QueueReadTxBuf)',
    r'DQGrou(p\d\.QueueReadPairBuf)',
    r'DQGrou(p\d\.QueueWritePairBuf)',
    r'DQGrou(p\d\.CombWKNet)',
    r'DQGrou(p\d\.CombFWNet)',
    r'DQGrou(p\d\.CombSelNet)',
    r'DQGrou(p\d\.SRAMWritePointer)',
    r'DQGrou(p\d\.DQBank\d\.SRAMWriteInst)',
    r'DQGrou(p\d\.DQBank\d\.SRAMReadInst)',
    r'DQGrou(p\d\.DQBank\d\.SRAMReadPointer)',
    r'DQGrou(p\d\.DQBank\d\.SRAMWriteValue)',
    r'DQGrou(p\d\.DQBank\d\.SRAMReadValue)',
    r'DQGrou(p\d\.DQBank\d\.RegWriteValid)',
    r'DQGrou(p\d\.DQBank\d\.RegReadValid)',
    r'DQGrou(p\d\.DQBank\d\.RegWriteNbusy)',
    r'DQGrou(p\d\.DQBank\d\.RegReadNbusy)',
    r'DQGrou(p\d\.DQBank\d\.RegWriteRxBuf)',
    r'DQGrou(p\d\.DQBank\d\.RegReadRxBuf)',
    r'DQGrou(p\d\.DQBank\d\.QueueReadReadyInstBuf)',
    r'DQGrou(p\d\.DQBank\d\.QueueWriteReadyInstBuf)',
    r'num(Cycles)',
    r'(sim_seconds)',
    r'cpus\.(int_regfile_reads)',
    r'cpus\.(int_regfile_writes)',
    r'cpus\.(fp_regfile_reads)',
    r'cpus\.(fp_regfile_writes)',
    r'cpus\.(misc_regfile_reads)',
    r'cpus\.(misc_regfile_writes)',
    r'cpus\.iq\.(int_inst_queue_reads)',
    r'cpus\.iq\.(int_inst_queue_writes)',
    r'cpus\.iq\.(int_inst_queue_wakeup_accesses)',
    r'cpus\.iq\.(fp_inst_queue_reads)',
    r'cpus\.iq\.(fp_inst_queue_writes)',
    r'cpus\.iq\.(fp_inst_queue_wakeup_accesses)',
]
fu_targets= [
        'system\.switch_cpus\.commit\.op_class_0::(IntDiv)',
        'system\.switch_cpus\.commit\.op_class_0::(FloatDiv)',
        #'system\.switch_cpus\.commit\.op_class_0::(IntMult)',
        #'system\.switch_cpus\.commit\.op_class_0::(FloatMult)',
        #'system\.switch_cpus\.commit\.op_class_0::(FloatMultAcc)',
        ]

beta_targets = [
    'cpus?\.(ipc)',
    'cpus?\.committed(Insts)',
    '(l2\.demand_miss)es::total',
    'branchPred\.(condIncorrect)',
    'branchPred\.(indirectMispredicted)',
    'cpus?\.(dcache\.demand_misses)::total',
        ]

xs_ipc_target = {
    "commitInstr": r"\[PERF \]\[time=\s+\d+\] TOP.SimTop.l_soc.core_with_l2.core.ctrlBlock.rob: commitInstr,\s+(\d+)",
    "clock_cycle": r"\[PERF \]\[time=\s+\d+\] TOP.SimTop.l_soc.core_with_l2.core.ctrlBlock.rob: clock_cycle,\s+(\d+)",
}

xs_branch_targets = {
    'BpInstr':  r"\[PERF \]\[time=\s+\d+\] TOP\.SimTop\.l_soc\.core_with_l2\.core\.frontend\.ftq: BpInstr,\s+(\d+)",
    'BpBWrong': r"\[PERF \]\[time=\s+\d+\] TOP\.SimTop\.l_soc\.core_with_l2\.core\.frontend\.ftq: BpBWrong,\s+(\d+)",
    'BpJWrong': r"\[PERF \]\[time=\s+\d+\] TOP\.SimTop\.l_soc\.core_with_l2\.core\.frontend\.ftq: BpJWrong,\s+(\d+)",
    'BpIWrong': r"\[PERF \]\[time=\s+\d+\] TOP\.SimTop\.l_soc\.core_with_l2\.core\.frontend\.ftq: BpIWrong,\s+(\d+)",
    # 'BpBInstr': r"\[PERF \]\[time=\s+\d+\] TOP\.SimTop\.l_soc\.core_with_l2\.core\.frontend\.ftq: BpBInstr,\s+(\d+)",
    # 'BpRight':  r"\[PERF \]\[time=\s+\d+\] TOP\.SimTop\.l_soc\.core_with_l2\.core\.frontend\.ftq: BpRight,\s+(\d+)",
    # 'BpWrong':  r"\[PERF \]\[time=\s+\d+\] TOP\.SimTop\.l_soc\.core_with_l2\.core\.frontend\.ftq: BpWrong,\s+(\d+)",
    # 'BpBRight': r"\[PERF \]\[time=\s+\d+\] TOP\.SimTop\.l_soc\.core_with_l2\.core\.frontend\.ftq: BpBRight,\s+(\d+)",
    # 'BpJRight': r"\[PERF \]\[time=\s+\d+\] TOP\.SimTop\.l_soc\.core_with_l2\.core\.frontend\.ftq: BpJRight,\s+(\d+)",
    # 'BpIRight': r"\[PERF \]\[time=\s+\d+\] TOP\.SimTop\.l_soc\.core_with_l2\.core\.frontend\.ftq: BpIRight,\s+(\d+)",
    # 'BpCRight': r"\[PERF \]\[time=\s+\d+\] TOP\.SimTop\.l_soc\.core_with_l2\.core\.frontend\.ftq: BpCRight,\s+(\d+)",
    # 'BpCWrong': r"\[PERF \]\[time=\s+\d+\] TOP\.SimTop\.l_soc\.core_with_l2\.core\.frontend\.ftq: BpCWrong,\s+(\d+)",
    # 'BpRRight': r"\[PERF \]\[time=\s+\d+\] TOP\.SimTop\.l_soc\.core_with_l2\.core\.frontend\.ftq: BpRRight,\s+(\d+)",
    # 'BpRWrong': r"\[PERF \]\[time=\s+\d+\] TOP\.SimTop\.l_soc\.core_with_l2\.core\.frontend\.ftq: BpRWrong,\s+(\d+)",

    # '(ubtbRight)',
    # '(ubtbWrong)',
    # '(btbRight)',
    # '(btbWrong)',
    # '(tageRight)',
    # '(tageWrong)',
    # '(rasRight)',
    # '(rasWrong)',
    # '(loopRight)',
    # '(loopWrong)',
}

xs_cache_targets_22_04_nanhu = {
    'l3_acc': (r"\[PERF \]\[time=\s+\d+\] TOP\.SimTop\.l_soc\.l3cacheOpt: selfdir_A_req,\s+(\d+)", 4),
    'l3_hit': (r"\[PERF \]\[time=\s+\d+\] TOP\.SimTop\.l_soc\.l3cacheOpt: selfdir_A_hit,\s+(\d+)", 4),

    'l2_acc': (r"\[PERF \]\[time=\s+\d+\] TOP\.SimTop\.l_soc\.core_with_l2\.l2cache: selfdir_A_req,\s+(\d+)", 4),
    'l2_hit': (r"\[PERF \]\[time=\s+\d+\] TOP\.SimTop\.l_soc\.core_with_l2\.l2cache: selfdir_A_hit,\s+(\d+)", 4),

    'dcache_ammp': (r"dcache.missQueue.entries_\d+: load_miss_penalty_to_use,\s+(\d+)", 16),
}

xs_cache_targets_nanhu = {
}

def add_nanhu_l1_dcache_targets():
    for load_pipeline in range(2):
        xs_cache_targets_nanhu['l1d_{}_miss'.format(load_pipeline)] = r"\[PERF \]\[time=\s+\d+\] TOP\.SimTop\.l_soc\.core_with_l2\.core\.memBlock\.LoadUnit_{}.load_s2: dcache_miss_first_issue,\s+(\d+)".format(load_pipeline)
        xs_cache_targets_nanhu['l1d_{}_acc'.format(load_pipeline)] = r"\[PERF \]\[time=\s+\d+\] TOP\.SimTop\.l_soc\.core_with_l2\.core\.memBlock\.LoadUnit_{}.load_s2: in_fire_first_issue,\s+(\d+)".format(load_pipeline)


def add_nanhu_l2_l3_targets():
    for bank in range(4):
        xs_cache_targets_nanhu['l3b{}_acc'.format(bank)] = r"\[PERF \]\[time=\s+\d+\] TOP\.SimTop\.l_soc\.l3cacheOpt\.slices_{}.directory: selfdir_A_req,\s+(\d+)".format(bank)
        xs_cache_targets_nanhu['l3b{}_hit'.format(bank)] = r"\[PERF \]\[time=\s+\d+\] TOP\.SimTop\.l_soc\.l3cacheOpt\.slices_{}.directory: selfdir_A_hit,\s+(\d+)".format(bank)
        xs_cache_targets_nanhu['l3b{}_recv_pref'.format(bank)] = r"\[PERF \]\[time=\s+\d+\] TOP\.SimTop\.l_soc\.l3cacheOpt\.slices_{}\.a_req_buffer: recv_prefetch,\s+(\d+)".format(bank)

        xs_cache_targets_nanhu['l2b{}_acc'.format(bank)] = r"\[PERF \]\[time=\s+\d+\] TOP\.SimTop\.l_soc\.core_with_l2\.l2cache\.slices_{}.directory: selfdir_A_req,\s+(\d+)".format(bank)
        xs_cache_targets_nanhu['l2b{}_hit'.format(bank)] = r"\[PERF \]\[time=\s+\d+\] TOP\.SimTop\.l_soc\.core_with_l2\.l2cache\.slices_{}.directory: selfdir_A_hit,\s+(\d+)".format(bank)
        xs_cache_targets_nanhu['l2b{}_recv_pref'.format(bank)] = r"\[PERF \]\[time=\s+\d+\] TOP\.SimTop\.l_soc\.core_with_l2\.l2cache\.slices_{}\.a_req_buffer: recv_prefetch,\s+(\d+)".format(bank)

add_nanhu_l1_dcache_targets()
add_nanhu_l2_l3_targets()

xgroup_targets = [
    'cpus?\.num(Load)Insts',
    'cpus?\.(inter)_num',
    'cpus?\.(intra)_num',
    ]

