import pandas as pd
from copy import deepcopy


gem5_coarse_rename_map = {
    'NoStall': 'MergeBase',

    # Core
    'LongExecute': 'MergeCore',
    'InstNotReady': 'MergeCore',

    # Memory
    'LoadL1Bound': 'MergeLoad',
    'LoadL2Bound': 'MergeLoad',
    'LoadL3Bound': 'MergeLoad',
    'LoadMemBound': 'MergeLoad',
    'StoreL1Bound': 'MergeStore',
    'StoreL2Bound': 'MergeStore',
    'StoreL3Bound': 'MergeStore',
    'StoreMemBound': 'MergeStore',
    'DTlbStall': 'MergeLoad',

    # Frontend
    'IcacheStall': 'MergeFrontend',
    'ITlbStall': 'MergeFrontend',
    'FragStall': 'MergeFrontend',

    # BP
    'BpStall': 'MergeBadSpec',
    'SquashStall': 'MergeBadSpec',
    'InstMisPred': 'MergeBadSpec',
    'InstSquashed': 'MergeBadSpec',
    # BP + backend
    'CommitSquash': 'MergeBadSpec',

    # Unclassified:
    'SerializeStall': 'MergeMisc',
    'TrapStall': 'MergeMisc',
    'IntStall': 'MergeMisc',
    'ResumeUnblock': 'MergeMisc',
    'FetchBufferInvalid': 'MergeFrontend',
    'OtherStall': 'MergeMisc',
    'OtherFetchStall': 'MergeFrontend',
}
xs_coarse_rename_map = {
    'OverrideBubble': 'MergeFrontend',
    'FtqFullStall': 'MergeFrontend',
    'FtqUpdateBubble': 'MergeBadSpec',
    'TAGEMissBubble': 'MergeBadSpec',
    'SCMissBubble': 'MergeBadSpec',
    'ITTAGEMissBubble': 'MergeBadSpec',
    'RASMissBubble': 'MergeBadSpec',
    'ICacheMissBubble': 'MergeFrontend',
    'ITLBMissBubble': 'MergeFrontend',
    'BTBMissBubble': 'MergeBadSpec',
    'FetchFragBubble': 'MergeFrontend',

    'DivStall': 'MergeCore',
    'IntNotReadyStall': 'MergeCore',
    'FPNotReadyStall': 'MergeCore',

    'MemNotReadyStall': 'MergeLoad',

    'LoadTLBStall': 'MergeLoad',
    'LoadL1Stall': 'MergeLoad',
    'LoadL2Stall': 'MergeLoad',
    'LoadL3Stall': 'MergeLoad',
    'LoadMemStall': 'MergeLoad',
    'StoreStall': 'MergeStore',

    'AtomicStall': 'MergeMisc',

    'FlushedInsts': 'MergeBadSpec',
    'LoadVioReplayStall': 'MergeBadSpec',

    'LoadMSHRReplayStall': 'MergeLoad',

    'ControlRecoveryStall': 'MergeBadSpec',
    'MemVioRecoveryStall': 'MergeBadSpec',
    'OtherRecoveryStall': 'MergeBadSpec',
    
    'OtherCoreStall': 'MergeCore',
    'NoStall': 'MergeBase',

    'MemVioRedirectBubble': 'MergeBadSpec',
    'OtherRedirectBubble': 'MergeMisc',

}

xs_mem_finegrain_rename_map = deepcopy(xs_coarse_rename_map)

xs_mem_finegrain_rename_map.update({
    'MemNotReadyStall': 'MemNotReady',
    'LoadTLBStall': 'DTlbStall',
    'LoadL1Stall': 'LoadL1Bound',
    'LoadL2Stall': 'LoadL2Bound',
    'LoadL3Stall': 'LoadL3Bound',
    'LoadMemStall': 'LoadMemBound',
    'StoreStall': 'MergeStoreBound',
})

gem5_fine_grain_rename_map = {
    'NoStall': None,

    # Core
    'LongExecute': None,
    'InstNotReady': None,

    # Memory
    'LoadL1Bound': None,
    'LoadL2Bound': None,
    'LoadL3Bound': None,
    'LoadMemBound': None,
    'StoreL1Bound': 'MergeStoreBound',
    'StoreL2Bound': 'MergeStoreBound',
    'StoreL3Bound': 'MergeStoreBound',
    'StoreMemBound': 'MergeStoreBound',
    'DTlbStall': None,

    # Frontend
    'IcacheStall': 'ICacheBubble',
    'ITlbStall': 'ITlbBubble',
    'FragStall': 'FragmentBubble',

    # BP
    'BpStall': 'MergeBadSpecBubble',
    'SquashStall': 'MergeBadSpecBubble',
    'InstMisPred': 'MergeBadSpecBubble',
    'InstSquashed': 'BadSpecInst',

    # BP + backend
    'CommitSquash': 'BadSpecWalking',

    # Unclassified:
    'SerializeStall': None,
    'TrapStall': 'MergeMisc',
    'IntStall': 'MergeMisc',
    'ResumeUnblock': 'MergeMisc',
    'FetchBufferInvalid': 'MergeOtherFrontend',
    'OtherStall': 'MergeMisc',
    'OtherFetchStall': 'MergeOtherFrontend',
}

xs_fine_grain_rename_map = {
    'OverrideBubble': 'MergeOtherFrontend',
    'FtqFullStall': 'MergeOtherFrontend',
    'FtqUpdateBubble': 'MergeBadSpecBubble',
    'TAGEMissBubble': 'MergeBadSpecBubble',
    'SCMissBubble': 'MergeBadSpecBubble',
    'ITTAGEMissBubble': 'MergeBadSpecBubble',
    'RASMissBubble': 'MergeBadSpecBubble',
    'ICacheMissBubble': 'ICacheBubble',
    'ITLBMissBubble': 'ITlbBubble',
    'BTBMissBubble': 'MergeBadSpecBubble',
    'FetchFragBubble': 'FragmentBubble',

    'DivStall': 'LongExecute',
    'IntNotReadyStall': 'MergeInstNotReady',
    'FPNotReadyStall': 'MergeInstNotReady',

    'MemNotReadyStall': 'MemNotReady',

    'LoadTLBStall': 'DTlbStall',
    'LoadL1Stall': 'LoadL1Bound',
    'LoadL2Stall': 'LoadL2Bound',
    'LoadL3Stall': 'LoadL3Bound',
    'LoadMemStall': 'LoadMemBound',
    'StoreStall': 'MergeStoreBound',

    'AtomicStall': 'SerializeStall',

    'FlushedInsts': 'BadSpecInst',
    'LoadVioReplayStall': None,

    'LoadMSHRReplayStall': None,

    'ControlRecoveryStall': 'MergeBadSpecWalking',
    'MemVioRecoveryStall': 'MergeBadSpecWalking',
    'OtherRecoveryStall': 'MergeBadSpecWalking',
    
    'OtherCoreStall': 'MergeMisc',
    'commitInstr': None,

    'MemVioRedirectBubble': 'MergeBadSpecBubble',
    'OtherRedirectBubble': 'MergeMisc',

    'commitInstr': 'Insts',
    'total_cycles': 'Cycles',
}

def rename_with_map(df: pd.DataFrame, rename_map):
    to_drops = []
    sorted_cols = []
    print(df.columns)
    for col in df.columns:
        if col not in rename_map:
            to_drops.append(col)
    for k in rename_map:
        if rename_map[k] is not None:
            if rename_map[k].startswith('Merge'):
                merged = rename_map[k][5:]
                if merged not in df.columns:
                    df[merged] = df[k]
                    sorted_cols.append(merged)
                else:
                    df[merged] += df[k]
            else:
                df[rename_map[k]] = df[k]
                sorted_cols.append(rename_map[k])

            to_drops.append(k)
        else:
            sorted_cols.append(k)
    print(f'Dropping {to_drops}')
    df.drop(columns=to_drops, inplace=True)
