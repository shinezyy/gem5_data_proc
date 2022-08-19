brief_targets = [
    'cpus?\.(ipc)',
    'cpus?\.(cpi)',
    'cpus?\.committed(Insts)',
    'cpus?\.(lastCommitTick)',
    'host_(inst_rate)',
    #'cpus\.num(Cycles)'
]
ipc_target = [
    'cpus?\.(ipc)',
    'cpus?\.(cpi)',
]

flow_target = [
    'DQGroup\d\.(WKFlowUsage::\d+)',
    'DQGroup\d\.(WKFlowUsage::\w+)',
        ]

standard_targets = [
    '(numCycles)',
    'cpus?\.committed(Insts)',
    'cpus?\.(ipc)',
]

cache_targets = [
    # '(l3\.demandMissRate)::total',
    '(l3\.demandMisses)::total',
    # '(l2\.demandMissRate)::total',
    '(l2\.demandMisses)::total',
    # 'cpu\.(dcache\.demandAvgMissLatency)::\.cpu\.data',
    # 'cpu\.(dcache\.demandMisses)::\.cpu\.data',
    # 'cpu\.(dcache.demandMissRate)::\.cpu\.data',
    'cpu\.(dcache\.overallAccesses)::cpu\.data',
    '(l2\.overallAccesses)::total',
    # 'cpu\.(icache.*_miss_rate)::\.cpus\.data',
    # 'cpu\.iew\.iew(ExecLoadInsts)',
]

branch_targets = [
    'cpus?\.(?:diewc|commit|iew)\.(branchMispredicts)',
    'cpus?\.(?:diewc\.exec_|commit\.)(branches)',
    'cpus?\.branchPred\.(indirectMispred)icted',
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

xs_ipc_target = [
    '(ipc)',
    '(totalCycle)',
    '(roq: commitInstr)',
]

xs_branch_targets = [
    '(BpInstr)',
    '(BpBInstr)',
    '(BpRight)',
    '(BpWrong)',
    '(BpBRight)',
    '(BpBWrong)',
    '(BpJRight)',
    '(BpJWrong)',
    '(BpIRight)',
    '(BpIWrong)',
    '(BpCRight)',
    '(BpCWrong)',
    '(BpRRight)',
    '(BpRWrong)',

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
]

xgroup_targets = [
    'cpus?\.num(Load)Insts',
    'cpus?\.(inter)_num',
    'cpus?\.(intra)_num',
    ]

