brief_targets = [
    'cpus\.(ipc)',
    'cpus\.committed(Insts)',
    #'cpus\.num(Cycles)'
]
ipc_target = [
    'cpus\.(ipc)',
]

flow_target = [
        'dataflow_queue\.(WKFlowUsage::\d+)',
        'dataflow_queue\.(WKFlowUsage::\w+)',
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
        'diewc\.(queueingD)elay',
        'diewc\.(ssrD)elay',
        'diewc\.(pendingD)elay',
        'diewc\.FU(ContentionD)elay',
        'dataflow_queue\.(HalfSquash)es',
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
