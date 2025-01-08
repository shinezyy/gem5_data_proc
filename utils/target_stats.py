import os
import copy

brief_targets = {
    'ipc': '(?:cpus?|switch_cpus_1)\.ipc',
    'Insts': '(?:cpus?|switch_cpus_1)\.committedInsts',
    'Cycles': 'cpus?\.numCycles',
}

ipc_target = {
    'ipc': '(?:cpus?|switch_cpus_1)\.ipc',
    'cpi': '(?:cpus?|switch_cpus_1)\.cpi',
}

sim_targets = {
    'sim_seconds': 'sim_seconds',
}

icache_targets = {
    'icache_miss': 'icache\.demandMisses::total',
    'icache_acc': 'icache\.overallAccesses::total',
}

cache_targets = {
    'l3_acc_l2_pref': 'l3\.demandAccesses::l2\.prefetcher',
    'l3_acc_l1_pref': 'l3\.demandAccesses::cpu\.dcache\.prefetcher',
    'l3_acc': 'l3\.demandAccesses::total',

    'l3_miss_l1d_pref': 'l3\.demandMisses::cpu\.dcache\.prefetcher',
    'l3_miss_l2_pref': 'l3\.demandMisses::l2\.prefetcher',
    'l3_miss': 'l3\.demandMisses::total',

    'l2_acc': 'l2_caches\.demandAccesses::total',
    'l2_miss': 'l2_caches\.demandMisses::total',
    'l2_miss_l1d_pref': 'l2_caches\.demandMisses::cpu\.dcache\.prefetcher',
    
    'dcache_miss': 'cpu\.dcache\.demandMisses::total',
    # 'dcache_miss_pref': 'cpu\.dcache\.demandMisses::cpu.dcache.prefetcher',
}

si_targets = {
    'SerialInsts': 'system\.cpu\.iew\.dispatchStallReason::SerializeStall',
}

mem_dep_targets = {
    'depLoads': '(?:cpus?|switch_cpus_1)?\.MemDepUnit__0.dependentLoads',
    'confLoads': '(?:cpus?|switch_cpus_1)?\.MemDepUnit__0.conflictingLoads',
    'confStores': '(?:cpus?|switch_cpus_1)?\.MemDepUnit__0.conflictingStores',
}

bank_conf_targets = {
    'bankConfs': '(?:cpus?|switch_cpus_1)?\.lsq0\.bankConflicts',
    'bankConfMisPredRate': '(?:cpus?|switch_cpus_1)?\.lsq0\.bankConflictMisPredRate',
    'bankConfMisPreds': '(?:cpus?|switch_cpus_1)?\.lsq0\.misPredictedBankConflicts',
}

mem_targets = {
    'Sec': 'simSeconds',
    'WritebackDirty': 'system.membus\.transDist::WritebackDirty',
    'ReadResp': 'system.membus\.transDist::ReadResp',
    'ReadExResp': 'system.membus\.transDist::ReadExResp',
}

pf_targets = {
    'l1_pf_unused': 'system\.(?:cpus?|switch_cpus_1)?\.dcache\.prefetcher\.(pfUnused) ',
    'l1_pf_useful': 'system\.(?:cpus?|switch_cpus_1)?\.dcache\.prefetcher\.(pfUseful) ',
    'l2_pf_unused': 'system\.l2\.prefetcher\.(pfUnused) ',
    'l2_pf_useful': 'system\.l2\.prefetcher\.(pfUseful) ',
}

topdown_targets = {}

LievenStalls = [
    'NoStall',
    'IcacheStall',
    'ITlbStall',
    'DTlbStall',
    'BpStall',
    'IntStall',
    'TrapStall',
    'FetchFragStall',
    'OtherFragStall',
    'FTQBubble',
    'SquashStall',
    'FetchBufferInvalid',
    'InstMisPred',
    'InstSquashed',
    'SerializeStall',
    'ScalarLongExecute',
    'VectorLongExecute',
    'InstNotReady',

    'LoadL1Bound',
    'LoadL2Bound',
    'LoadL3Bound',
    'LoadMemBound',
    'StoreL1Bound',
    'StoreL2Bound',
    'StoreL3Bound',
    'StoreMemBound',
    'MemSquashed',
    'MemNotReady',
    'MemCommitRateLimit',
    'Atomic',
    'OtherMemStall',

    'MemDQBandwidth',
    'IntDQBandwidth',
    'FVDQBandwidth',
    'VectorReadyButNotIssued',
    'ScalarReadyButNotIssued',
    'ResumeUnblock',
    'CommitSquash',
    'OtherStall',
    'OtherFetchStall',
    ]

def add_topdown_targets():
    for stall in LievenStalls:
        topdown_targets[stall] = r'system\.cpu\.iew\.dispatchStallReason::{}'.format(stall)

add_topdown_targets()

warmup_targets = {
    'branchMispredicts': '(?:cpus?|switch_cpus_1)\.commit\.branchMispredicts',
    'branches': '(?:cpus?|switch_cpus_1)\.commit\.branches',
    'l3_miss': 'l3\.demandMisses::total',
    'l2_miss': 'l2\.demandMisses::total',
    'numCycles': 'numCycles',
    'Insts': '(?:cpus?|switch_cpus_1)?\.committedInsts',
}

branch_targets = {
    'branchMispredicts': '(?:cpus?|switch_cpus_1)\.commit\.branchMispredicts',
    'branches': '(?:cpus?|switch_cpus_1)\.commit\.branches',
    'indirectMispred': '(?:cpus?|switch_cpus_1)\.branchPred\.ftb\.indirectPredCorrect',
}


xs_l3_prefix = "\[PERF \]\[time=\s+\d+\] SimTop\.l_soc\.l3cacheOpt"

if 'XS_CORE_ID' not in os.environ or int(os.environ['XS_CORE_ID']) == 0:
    xs_core_prefix = "\[PERF \]\[time=\s+\d+\] SimTop\.l_soc\.core_with_l2\.core"
    xs_ctrl_block_prefix = "\[PERF \]\[time=\s+\d+\] SimTop\.l_soc\.core_with_l2\.core\.(?:backend\.)?inner.ctrlBlock"
    xs_l2_prefix = "\[PERF \]\[time=\s+\d+\] SimTop\.l_soc\.core_with_l2\.l2top\.inner.l2cache"
elif int(os.environ['XS_CORE_ID']) > 0:
    cur_core_id = int(os.environ['XS_CORE_ID'])
    xs_core_prefix = f"\[PERF \]\[time=\s+\d+\] SimTop\.l_soc\.core_with_l2_{cur_core_id}\.core"
    xs_ctrl_block_prefix = f"\[PERF \]\[time=\s+\d+\] SimTop\.l_soc\.core_with_l2_{cur_core_id}\.core\.(?:backend\.)?inner.ctrlBlock"
    xs_l2_prefix = f"\[PERF \]\[time=\s+\d+\] SimTop\.l_soc\.core_with_l2_{cur_core_id}\.l2top\.inner.l2cache"

xs_pf_targets = {
    'sms_useful': f'{xs_l2_prefix}\.topDown: L2prefetchUsefulSMS,\s+(\d+)',
    'sms_sent': f'{xs_l2_prefix}\.topDown: L2prefetchSentSMS,\s+(\d+)',
}

xs_ipc_target = {
    "commitInstr": fr"{xs_ctrl_block_prefix}.rob: commitInstr,\s+(\d+)",
    "total_cycles": fr"{xs_ctrl_block_prefix}.rob: clock_cycle,\s+(\d+)",
}

xs_mem_dep_targets = {
    'depLoads': fr'{xs_ctrl_block_prefix}\.dispatch: storeset_load_wait,\s+(\d+)',
    'confLoads': fr'{xs_core_prefix}\.memBlock.lsq.loadQueue.loadQueueRAW: stld_rollback,\s+(\d+)',
}

# align
xs_topdown_targets = {
    'NoStall': fr'{xs_ctrl_block_prefix}\.dispatch: NoStall,\s+(\d+)',
    'OverrideBubble': fr'{xs_ctrl_block_prefix}\.dispatch: OverrideBubble,\s+(\d+)',
    'FtqUpdateBubble': fr'{xs_ctrl_block_prefix}\.dispatch: FtqUpdateBubble,\s+(\d+)',
    'TAGEMissBubble': fr'{xs_ctrl_block_prefix}\.dispatch: TAGEMissBubble,\s+(\d+)',
    'SCMissBubble': fr'{xs_ctrl_block_prefix}\.dispatch: SCMissBubble,\s+(\d+)',
    'ITTAGEMissBubble': fr'{xs_ctrl_block_prefix}\.dispatch: ITTAGEMissBubble,\s+(\d+)',
    'RASMissBubble': fr'{xs_ctrl_block_prefix}\.dispatch: RASMissBubble,\s+(\d+)',
    'MemVioRedirectBubble': fr'{xs_ctrl_block_prefix}\.dispatch: MemVioRedirectBubble,\s+(\d+)',
    'OtherRedirectBubble': fr'{xs_ctrl_block_prefix}\.dispatch: OtherRedirectBubble,\s+(\d+)',
    'FtqFullStall': fr'{xs_ctrl_block_prefix}\.dispatch: FtqFullStall,\s+(\d+)',
    'ICacheMissBubble': fr'{xs_ctrl_block_prefix}\.dispatch: ICacheMissBubble,\s+(\d+)',
    'ITLBMissBubble': fr'{xs_ctrl_block_prefix}\.dispatch: ITLBMissBubble,\s+(\d+)',
    'BTBMissBubble': fr'{xs_ctrl_block_prefix}\.dispatch: BTBMissBubble,\s+(\d+)',
    'FetchFragBubble': fr'{xs_ctrl_block_prefix}\.dispatch: FetchFragBubble,\s+(\d+)',
    'DivStall': fr'{xs_ctrl_block_prefix}\.dispatch: DivStall,\s+(\d+)',
    'IntNotReadyStall': fr'{xs_ctrl_block_prefix}\.dispatch: IntNotReadyStall,\s+(\d+)',
    'FPNotReadyStall': fr'{xs_ctrl_block_prefix}\.dispatch: FPNotReadyStall,\s+(\d+)',
    'MemNotReadyStall': fr'{xs_ctrl_block_prefix}\.dispatch: MemNotReadyStall,\s+(\d+)',
    'LoadTLBStall': fr'{xs_ctrl_block_prefix}\.dispatch: LoadTLBStall,\s+(\d+)',
    'LoadL1Stall': fr'{xs_ctrl_block_prefix}\.dispatch: LoadL1Stall,\s+(\d+)',
    'LoadL2Stall': fr'{xs_ctrl_block_prefix}\.dispatch: LoadL2Stall,\s+(\d+)',
    'LoadL3Stall': fr'{xs_ctrl_block_prefix}\.dispatch: LoadL3Stall,\s+(\d+)',
    'LoadMemStall': fr'{xs_ctrl_block_prefix}\.dispatch: LoadMemStall,\s+(\d+)',
    'StoreStall': fr'{xs_ctrl_block_prefix}\.dispatch: StoreStall,\s+(\d+)',
    'AtomicStall': fr'{xs_ctrl_block_prefix}\.dispatch: AtomicStall,\s+(\d+)',
    'LoadVioReplayStall': fr'{xs_ctrl_block_prefix}\.dispatch: LoadVioReplayStall,\s+(\d+)',
    'LoadMSHRReplayStall': fr'{xs_ctrl_block_prefix}\.dispatch: LoadMSHRReplayStall,\s+(\d+)',
    'ControlRecoveryStall': fr'{xs_ctrl_block_prefix}\.dispatch: ControlRecoveryStall,\s+(\d+)',
    'MemVioRecoveryStall': fr'{xs_ctrl_block_prefix}\.dispatch: MemVioRecoveryStall,\s+(\d+)',
    'OtherRecoveryStall': fr'{xs_ctrl_block_prefix}\.dispatch: OtherRecoveryStall,\s+(\d+)',
    'FlushedInsts': fr'{xs_ctrl_block_prefix}\.dispatch: FlushedInsts,\s+(\d+)',
    'OtherCoreStall': fr'{xs_ctrl_block_prefix}\.dispatch: OtherCoreStall,\s+(\d+)',
}

# old
xs_topdown_targets_deprecated = {
    'fetch_bubbles':  fr"{xs_core_prefix}ctrlBlock.decode: fetch_bubbles,\s+(\d+)",
    'decode_bubbles': fr"{xs_core_prefix}ctrlBlock.decode: decode_bubbles,\s+(\d+)",
    "slots_issued":   fr"{xs_core_prefix}ctrlBlock.decode: slots_issued,\s+(\d+)",
    "recovery_bubbles": fr"{xs_core_prefix}ctrlBlock.rename: recovery_bubbles,\s+(\d+)",
    "slots_retired": fr"{xs_core_prefix}ctrlBlock.rob: commitUop,\s+(\d+)",
    "br_mispred_retired": fr"{xs_core_prefix}frontend.ftq: mispredictRedirect,\s+(\d+)",
    "icache_miss_cycles": fr"{xs_core_prefix}frontend.icache.mainPipe: icache_bubble_s2_miss,\s+(\d+)",
    "itlb_miss_cycles": fr"{xs_core_prefix}frontend.icache.mainPipe: icache_bubble_s0_tlb_miss,\s+(\d+)",
    "s2_redirect_cycles": fr"{xs_core_prefix}frontend.bpu: s2_redirect,\s+(\d+)",
    "s3_redirect_cycles": fr"{xs_core_prefix}frontend.bpu: s3_redirect,\s+(\d+)",
    "store_bound_cycles": fr"{xs_core_prefix}exuBlocks.scheduler: stall_stores_bound,\s+(\d+)",
    "load_bound_cycles": fr"{xs_core_prefix}exuBlocks.scheduler: stall_loads_bound,\s+(\d+)",
    "ls_dq_bound_cycles": fr"{xs_core_prefix}exuBlocks.scheduler: stall_ls_bandwidth_bound,\s+(\d+)",
    "stall_cycle_rob": fr"{xs_core_prefix}ctrlBlock.dispatch: stall_cycle_rob,\s+(\d+)",
    "stall_cycle_int_dq": fr"{xs_core_prefix}ctrlBlock.dispatch: stall_cycle_int_dq,\s+(\d+)",
    "stall_cycle_fp_dq": fr"{xs_core_prefix}ctrlBlock.dispatch: stall_cycle_fp_dq,\s+(\d+)",
    "stall_cycle_ls_dq": fr"{xs_core_prefix}ctrlBlock.dispatch: stall_cycle_ls_dq,\s+(\d+)",
    "stall_cycle_fp": fr"{xs_core_prefix}ctrlBlock.rename: stall_cycle_fp,\s+(\d+)",
    "stall_cycle_int": fr"{xs_core_prefix}ctrlBlock.rename: stall_cycle_int,\s+(\d+)",
    "l1d_loads_bound_cycles": fr"{xs_core_prefix}memBlock.lsq.loadQueue: l1d_loads_bound,\s+(\d+)",
    "l1d_loads_mshr_bound": fr"{xs_core_prefix}exuBlocks.scheduler.rs_3.loadRS_0: l1d_loads_mshr_bound,\s+(\d+)",
    "l1d_loads_tlb_bound": fr"{xs_core_prefix}exuBlocks.scheduler.rs_3.loadRS_0: l1d_loads_tlb_bound,\s+(\d+)",
    "l1d_loads_store_data_bound": fr"{xs_core_prefix}exuBlocks.scheduler.rs_3.loadRS_0: l1d_loads_store_data_bound,\s+(\d+)",
    "l1d_loads_bank_conflict_bound": fr"{xs_core_prefix}exuBlocks.scheduler.rs_3.loadRS_0: l1d_loads_bank_conflict_bound,\s+(\d+)",
    "l1d_loads_vio_check_redo_bound": fr"{xs_core_prefix}exuBlocks.scheduler.rs_3.loadRS_0: l1d_loads_vio_check_redo_bound,\s+(\d+)",
    "l2_loads_bound_cycles": fr"{xs_l2_prefix}: l2_loads_bound,\s+(\d+)",
    "l3_loads_bound_cycles": fr"{xs_l3_prefix}: l3_loads_bound,\s+(\d+)",
    "ddr_loads_bound_cycles": fr"{xs_l3_prefix}: ddr_loads_bound,\s+(\d+)",

    "stage2_redirect_cycles": fr"{xs_core_prefix}ctrlBlock: stage2_redirect_cycles,\s+(\d+)",
    "branch_resteers_cycles": fr"{xs_core_prefix}ctrlBlock: branch_resteers_cycles,\s+(\d+)",
    "robFlush_bubble_cycles": fr"{xs_core_prefix}ctrlBlock: robFlush_bubble_cycles,\s+(\d+)",
    "ldReplay_bubble_cycles": fr"{xs_core_prefix}ctrlBlock: ldReplay_bubble_cycles,\s+(\d+)",
    "ifu2id_allNO_cycle": fr"{xs_core_prefix}ctrlBlock.decode: ifu2id_allNO_cycle,\s+(\d+)",
}


xs_branch_targets = {
    'BpInstr':  r"\[PERF \]\[time=\s+\d+\] SimTop\.l_soc\.core_with_l2\.core\.frontend\.ftq: BpInstr,\s+(\d+)",
    'BpBWrong': r"\[PERF \]\[time=\s+\d+\] SimTop\.l_soc\.core_with_l2\.core\.frontend\.ftq: BpBWrong,\s+(\d+)",
    'BpJWrong': r"\[PERF \]\[time=\s+\d+\] SimTop\.l_soc\.core_with_l2\.core\.frontend\.ftq: BpJWrong,\s+(\d+)",
    'BpIWrong': r"\[PERF \]\[time=\s+\d+\] SimTop\.l_soc\.core_with_l2\.core\.frontend\.ftq: BpIWrong,\s+(\d+)",
    # 'BpBInstr': r"\[PERF \]\[time=\s+\d+\] SimTop\.l_soc\.core_with_l2\.core\.frontend\.ftq: BpBInstr,\s+(\d+)",
    # 'BpRight':  r"\[PERF \]\[time=\s+\d+\] SimTop\.l_soc\.core_with_l2\.core\.frontend\.ftq: BpRight,\s+(\d+)",
    # 'BpWrong':  r"\[PERF \]\[time=\s+\d+\] SimTop\.l_soc\.core_with_l2\.core\.frontend\.ftq: BpWrong,\s+(\d+)",
    # 'BpBRight': r"\[PERF \]\[time=\s+\d+\] SimTop\.l_soc\.core_with_l2\.core\.frontend\.ftq: BpBRight,\s+(\d+)",
    # 'BpJRight': r"\[PERF \]\[time=\s+\d+\] SimTop\.l_soc\.core_with_l2\.core\.frontend\.ftq: BpJRight,\s+(\d+)",
    # 'BpIRight': r"\[PERF \]\[time=\s+\d+\] SimTop\.l_soc\.core_with_l2\.core\.frontend\.ftq: BpIRight,\s+(\d+)",
    # 'BpCRight': r"\[PERF \]\[time=\s+\d+\] SimTop\.l_soc\.core_with_l2\.core\.frontend\.ftq: BpCRight,\s+(\d+)",
    # 'BpCWrong': r"\[PERF \]\[time=\s+\d+\] SimTop\.l_soc\.core_with_l2\.core\.frontend\.ftq: BpCWrong,\s+(\d+)",
    # 'BpRRight': r"\[PERF \]\[time=\s+\d+\] SimTop\.l_soc\.core_with_l2\.core\.frontend\.ftq: BpRRight,\s+(\d+)",
    # 'BpRWrong': r"\[PERF \]\[time=\s+\d+\] SimTop\.l_soc\.core_with_l2\.core\.frontend\.ftq: BpRWrong,\s+(\d+)",

}

xs_cache_targets_22_04_nanhu = {
    'l3_acc': (r"\[PERF \]\[time=\s+\d+\] SimTop\.l_soc\.l3cacheOpt: selfdir_A_req,\s+(\d+)", 4),
    'l3_hit': (r"\[PERF \]\[time=\s+\d+\] SimTop\.l_soc\.l3cacheOpt: selfdir_A_hit,\s+(\d+)", 4),

    'l2_acc': (r"\[PERF \]\[time=\s+\d+\] SimTop\.l_soc\.core_with_l2\.l2cache: selfdir_A_req,\s+(\d+)", 4),
    'l2_hit': (r"\[PERF \]\[time=\s+\d+\] SimTop\.l_soc\.core_with_l2\.l2cache: selfdir_A_hit,\s+(\d+)", 4),

    'dcache_ammp': (r"dcache.missQueue.entries_\d+: load_miss_penalty_to_use,\s+(\d+)", 16),
}

xs_cache_targets_no_l3 = {}
xs_cache_targets = {}

def add_nanhu_l1_dcache_targets():
    for load_pipeline in range(2):
        xs_cache_targets['l1d_{}_miss'.format(load_pipeline)] = r"\[PERF \]\[time=\s+\d+\] SimTop\.l_soc\.core_with_l2\.core\.memBlock\.inner.LoadUnit_{}: s2_dcache_miss_first_issue,\s+(\d+)".format(load_pipeline)
        xs_cache_targets['l1d_{}_acc'.format(load_pipeline)] = r"\[PERF \]\[time=\s+\d+\] SimTop\.l_soc\.core_with_l2\.core\.memBlock\.inner.LoadUnit_{}: s2_in_fire_first_issue,\s+(\d+)".format(load_pipeline)
    for mshr in range(16):
        xs_cache_targets[f'l1d_mshr{mshr}_load_to_use_samples'] = rf"\[PERF \]\[time=\s+\d+\] SimTop\.l_soc\.core_with_l2\.core\.memBlock\.inner.dcache\.dcache\.missQueue\.entries_{mshr}: load_miss_penalty_to_use_sampled,\s+(\d+)"
        xs_cache_targets[f'l1d_mshr{mshr}_load_to_use'] = rf"\[PERF \]\[time=\s+\d+\] SimTop\.l_soc\.core_with_l2\.core\.memBlock\.inner.dcache\.dcache\.missQueue\.entries_{mshr}: load_miss_penalty_to_use,\s+(\d+)"


def add_nanhu_l2_targets():
    for bank in range(4):
        xs_cache_targets['l2b{}_acc'.format(bank)] = r"\[PERF \]\[time=\s+\d+\] SimTop\.l_soc\.core_with_l2\.l2cache\.slices_{}.directory: selfdir_A_req,\s+(\d+)".format(bank)
        xs_cache_targets['l2b{}_hit'.format(bank)] = r"\[PERF \]\[time=\s+\d+\] SimTop\.l_soc\.core_with_l2\.l2cache\.slices_{}.directory: selfdir_A_hit,\s+(\d+)".format(bank)
        xs_cache_targets['l2b{}_recv_pref'.format(bank)] = r"\[PERF \]\[time=\s+\d+\] SimTop\.l_soc\.core_with_l2\.l2cache\.slices_{}\.a_req_buffer: recv_prefetch,\s+(\d+)".format(bank)

xs_mem_acc_sources = ['CPUInst', 'CPULoadData', 'CPUStoreData', 'CPUAtomicData', 'L1InstPrefetch',
                      'L1DataPrefetch', 'PTW', 'Prefetch2L2BOP', 'Prefetch2L2PBOP', 'Prefetch2L2SMS',
                      'Prefetch2L2Stream', 'Prefetch2L2Stride', 'Prefetch2L2TP', 'Prefetch2L2Unknown',
                      'Prefetch2L3Unknown']

def add_kmh_l2_targets():
    for counter in ['Total', 'Miss']:
        for source in xs_mem_acc_sources:
            xs_cache_targets[f'l2_{source}_{counter}'] = fr"{xs_l2_prefix}\.topDown: E2_L2AReqSource_{source}_{counter},\s+(\d+)"
    
def add_xs_l3_targets():
    for counter in ['Total', 'Miss']:
        for source in xs_mem_acc_sources:
            xs_cache_targets[f'l3_{source}_{counter}'] = fr"{xs_l3_prefix}\.topDown: E2_L3AReqSource_{source}_{counter},\s+(\d+)"

    for bank in range(4):
        xs_cache_targets[f'l3_bank_{bank}_acc'] = fr"{xs_l3_prefix}\.slices_{bank}\.directory: selfdir_A_req,\s+(\d+)"
        xs_cache_targets[f'l3_bank_{bank}_hit'] = fr"{xs_l3_prefix}\.slices_{bank}\.directory: selfdir_A_hit,\s+(\d+)"
        # xs_cache_targets[f'l3_bank_{bank}_recv_pref'] = fr"{xs_l3_prefix}\.slices_{bank}\.a_req_buffer: recv_prefetch,\s+(\d+)"
        # xs_cache_targets[f'l3_bank_{bank}_recv_normal'] = fr"{xs_l3_prefix}\.slices_{bank}\.a_req_buffer: recv_normal,\s+(\d+)"


add_nanhu_l1_dcache_targets()
# add_nanhu_l2_targets()
add_kmh_l2_targets()
xs_cache_targets_no_l3 = copy.deepcopy(xs_cache_targets)
add_xs_l3_targets()

xs_mem_targets = {}

def add_xs_mem_bw_targets():
    xs_mem_targets['l3_bus_acq'] = r"\[PERF \]\[time=\s+\d+\] SimTop\.l_soc\." + \
            r"socMisc.busPMU: L3_Mem_L3_bank_0_D_channel_GrantData_fire,\s+(\d+)"
    xs_mem_targets['l3_bus_rel'] = r"\[PERF \]\[time=\s+\d+\] SimTop\.l_soc\." + \
            r"socMisc.busPMU: L3_Mem_L3_bank_0_C_channel_ReleaseData_fire,\s+(\d+)"

add_xs_mem_bw_targets()

def add_nanhu_multicore_ipc_targets(n):
    #add instr and clocks for other cores
    for core in range(1, n):
        xs_ipc_target[f'commitInstr{core}'] = r"\[PERF \]\[time=\s+\d+\] SimTop\.l_soc\.core_with_l2_{}.core\.ctrlBlock\.rob: commitInstr,\s+(\d+)".format(core)
        xs_ipc_target[f'total_cycles{core}'] = r"\[PERF \]\[time=\s+\d+\] SimTop\.l_soc\.core_with_l2_{}.core\.ctrlBlock\.rob: clock_cycle,\s+(\d+)".format(core)

rvv_targets = {
    'unitStrideCrossed': 'system\.cpu\.lsq0\.unitStrideCross16Byte',
    'segUnitStrideNF=4': 'system\.cpu\.commit\.segUnitStrideNF::4',
    'segUnitStrideNF=6': 'system\.cpu\.commit\.segUnitStrideNF::6',
}

for vec_inst_type in ['UnitStrideLoad', 'SegUnitStrideLoad', 'UnitStrideStore', 'SegUnitStrideStore',
                      'UnitStrideMaskLoad', 'SegUnitStrideMaskLoad' 'UnitStrideMaskStore',
                      'StridedLoad', 'SegStridedLoad', 'StridedStore',
                      'IndexedLoad', 'SegIndexedLoad', 'IndexedStore',
                      'UnitStrideFaultOnlyFirstLoad', 'WholeRegisterLoad', 'WholeRegisterStore',
                      'IntegerArith', 'FloatArith', 'FloatConvert', 'IntegerReduce', 'FloatReduce', 'Misc',
                      'IntegerExtension', 'Config']:
    rvv_targets[f'{vec_inst_type}'] = f"system\.cpu\.commit\.committedInstType_0::Vector{vec_inst_type}"
