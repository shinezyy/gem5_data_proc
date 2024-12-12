import pandas as pd
from copy import deepcopy


gem5_topdown_hierarchy = {
    'NoStall': ['NoStall'],
    'Frontend': ['IcacheStall', 'ITlbStall', 'FetchFragStall', 'FTQBubble', 'OtherFetchStall'],
    'Backend': {
        'Core': ['ScalarLongExecute', 'InstNotReady'],
        'Memory': {
            'Load': ['LoadL1Bound', 'LoadL2Bound', 'LoadL3Bound', 'LoadMemBound', 'DTlbStall'],
            'Store': ['StoreL1Bound'],
            'Other': ['MemNotReady', 'MemCommitRateLimit', 'OtherMemStall']
        },
        'Fragment': ['OtherFragStall'],
        'Dispatch': ['MemDQBandwidth', 'IntDQBandwidth', 'FVDQBandwidth', 'ScalarReadyButNotIssued'],
        'Misc': ['SerializeStall']
    },
    'BadSpec': {
        'Inst': ['BadSpecInst'],
        'Other': ['BpStall', 'SquashStall', 'InstSquashed', 'CommitSquash']
    },
}

gem5_coarse_rename_map = {
    'NoStall': 'MergeBase',

    # Core
    'ScalarLongExecute': 'MergeCore',
    'VectorLongExecute': 'MergeCore',
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
    'MemNotReady': 'MergeMem',
    'MemCommitRateLimit': 'MergeMem',

    # Frontend
    'IcacheStall': 'MergeFrontend',
    'ITlbStall': 'MergeFrontend',
    'FetchFragStall': 'MergeFrontend',
    'OtherFragStall': 'MergeFrontend',
    'FetchBufferInvalid': 'MergeFrontend',
    'OtherFetchStall': 'MergeFrontend',
    
    # BP
    'BpStall': 'MergeBadSpec',
    'SquashStall': 'MergeBadSpec',
    'InstMisPred': 'MergeBadSpec',
    'InstSquashed': 'MergeBadSpec',
    'CommitSquash': 'MergeBadSpec',

    # Unclassified:
    'SerializeStall': 'MergeMisc',
    'TrapStall': 'MergeMisc',
    'IntStall': 'MergeMisc',
    'ResumeUnblock': 'MergeMisc',
    'OtherStall': 'MergeMisc',
    
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
    'NoStall': 'MergeBase',

    # Core
    'ScalarLongExecute': None,
    'VectorLongExecute': None,
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
    'MemSquashed': 'MergeMem',
    'MemNotReady': 'MergeMem',
    'MemCommitRateLimit': 'MergeMem',

    # Frontend
    'IcacheStall': 'ICacheBubble',
    'ITlbStall': 'ITlbBubble',
    'FetchFragStall': 'FragmentBubble',
    'OtherFragStall': 'FragmentBubble',

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

    'DivStall': 'MergeLongExecute',
    'ScalarLongExecute': 'MergeLongExecute',
    'VectorLongExecute': 'MergeLongExecute',
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
    columns_to_keep = ['cpi', 'point', 'bmk', 'workload']  # 需要保留的列

    print(df.columns)
    for col in df.columns:
        if col not in rename_map and col not in columns_to_keep:
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
            if k not in columns_to_keep:
                to_drops.append(k)
        else:
            sorted_cols.append(k)
    print(f'Dropping {to_drops}')
    df.drop(columns=to_drops, inplace=True)
    
def create_rename_map(hierarchy, level=3, prefix=''):
    rename_map = {}
    for key, value in hierarchy.items():
        new_prefix = f"{prefix}{key}_" if prefix else key
        if isinstance(value, list):
            if level == 1 or (level == 2 and not prefix):
                rename_map.update({item: f'Merge{new_prefix}' for item in value})
            else:
                rename_map.update({item: None for item in value})
        elif isinstance(value, dict):
            if level == 1:
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, list):
                        rename_map.update({item: f'Merge{new_prefix}' for item in sub_value})
                    elif isinstance(sub_value, dict):
                        for sub_sub_key, sub_sub_value in sub_value.items():
                            rename_map.update({item: f'Merge{new_prefix}' for item in sub_sub_value})
            else:
                rename_map.update(create_rename_map(value, level-1, new_prefix))
    return rename_map

def mergeBadSpecInst(df):
    icount = 20*10**6
    df['BadSpecInst'] = df['NoStall'] - icount
    df['NoStall'] = icount

def rename_with_map(df: pd.DataFrame, hierarchy, level):
    mergeBadSpecInst(df)
    if level == 3:
        # 当 level=3 时，我们不进行任何重命名或合并
        return
    rename_map = create_rename_map(hierarchy, level)
    to_drops = []
    columns_to_keep = ['cpi', 'point', 'bmk', 'workload']

    for col in df.columns:
        if col not in rename_map and col not in columns_to_keep:
            to_drops.append(col)
    
    for k, v in rename_map.items():
        if v is not None:
            if v.startswith('Merge'):
                merged = v[5:]
                if merged not in df.columns:
                    df[merged] = df[k]
                else:
                    df[merged] += df[k]
            else:
                df[v] = df[k]
            if k not in columns_to_keep:
                to_drops.append(k)
    
    print(f'Dropping {to_drops}')
    df.drop(columns=to_drops, inplace=True)
