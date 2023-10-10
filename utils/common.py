import sys
from os.path import join as pjoin
from os.path import expanduser as expu
import re
import json
from copy import deepcopy
import pandas as pd

from local_configs import Env
from paths import *
from . import target_stats as t

env = Env()

def user_verify():
    ok = input('Is that OK? (y/n)')
    if ok != 'y':
        sys.exit()


def left_is_older(x, y):
    return os.path.getmtime(x) < os.path.getmtime(y)

# reverse a file

def reverse_readline(filename: str, buf_size=8192):
    """a generator that returns the lines of a file in reverse order"""
    with open(filename) as fh:
        segment = None
        offset = 0
        fh.seek(0, os.SEEK_END)
        file_size = remaining_size = fh.tell()
        while remaining_size > 0:
            offset = min(file_size, offset + buf_size)
            fh.seek(file_size - offset)
            buffer = fh.read(min(remaining_size, buf_size))
            remaining_size -= buf_size
            lines = buffer.split('\n')
            # the first line of the buffer is probably not a complete line so
            # we'll save it and append it to the last line of the next buffer
            # we read
            if segment is not None:
                # if the previous chunk starts right from the beginning of line
                # do not concact the segment to the last line of new chunk
                # instead, yield the segment first
                if buffer[-1] != '\n':
                    lines[-1] += segment
                else:
                    yield segment
            segment = lines[0]
            for index in range(len(lines) - 1, 0, -1):
                if len(lines[index]):
                    yield lines[index]
        # Don't yield None if the file was empty
        if segment is not None:
            yield segment


# scan pairs and filters:

def pair_to_dir(x, y):
    return x + '_' + y

def dir_to_pair(d):
    return d.split('_')

def pairs(stat_dir, return_path=True):
    # type: (string) -> list
    pairs_s = [x for x in os.listdir(expu(stat_dir))]
    pair_dirs = [pjoin(stat_dir, x) for x in pairs_s]
    if return_path:
        # return [x for x in pair_dirs if os.path.isdir(expu(x)) and '_' in x]
        return [x for x in pair_dirs if os.path.isdir(expu(x))]
    else:
        return [x[1] for x in zip(pair_dirs, pairs_s) \
                if os.path.isdir(expu(x[0]))]

def stat_filt(pairs, dirs, stat_name='stats.txt'):
    # type: (list) -> list
    new_pairs = []
    new_paths =[]
    for pair, path in zip(pairs, dirs):
        if os.path.isfile(expu(pjoin(path, stat_name))):
            new_pairs.append(pair)
            new_paths.append(path)
    return new_pairs, new_paths

def pair_to_full_path(path, pairs):
    return [expu(pjoin(path, p)) for p in pairs]

def pair_filt(pairs: list, fname: str):
    with open(fname) as f:
        selected_pairs = f.readlines()
        proc_pairs = []
        for p in selected_pairs:
            p = p.rstrip().replace(' ', '_')
            proc_pairs.append(p)
        new_list = []
        for p in pairs:
            if p in proc_pairs:
                new_list.append(p)
        return new_list


def time_filt(dirs):
    # type: (list) -> list
    def newer_than_gem5(d):
        stat = pjoin(expu(d), 'stats.txt')
        return left_is_older(pjoin(os.environ['gem5_build'], 'gem5.fast'), stat)
    return list(filter(newer_than_gem5, dirs))


# prints:

def print_list(l):
    cur_line_len = 0
    for x in l:
        if cur_line_len + len(str(x)) > 70:
            print('')
            cur_line_len = 0
        print(x, '\t', end=' ')
        cur_line_len += len(str(x))
    print('')


def print_option(opt):
    cur_line_len = 0
    for line in opt:
        if line.startswith('-') or cur_line_len + len(line) > 80:
            print('')
            cur_line_len = 0
        cur_line_len += len(line)
        print(line, end=' ')
    print('')


def print_2_tuple(x, y):
    x = str(x)
    if isinstance(y, float):
        y = '{:6f}'.format(y)
    else:
        y = str(y)
    print(x.ljust(25), y.rjust(20))

def print_dict(d):
    for k in d:
        print_2_tuple(k, d[k])


def print_line():
    print('---------------------------------------------------------------')


def extract_weight_from_config(config_file: str):
    assert osp.isfile(config_file)
    weight_pattern = re.compile('.*/{\w+}_\d+_(\d+\.\d+([eE][-+]?\d+)?)/.*\.gz')
    with open(config_file) as jf:
        js = json.load(jf)
        gcpt_file = js['system']['gcpt_file']
        m = weight_pattern.match(gcpt_file)
        assert m is not None
        weight = float(m.group(1))
        return weight


def extract_insts_from_config(config_file: str):
    assert osp.isfile(config_file)
    insts_pattern = re.compile(f'.*/_(\d+)_\.gz')
    with open(config_file) as jf:
        js = json.load(jf)
        gcpt_file = js['system']['gcpt_file']
        m = insts_pattern.match(gcpt_file)
        assert m is not None
        insts = int(m.group(1))
        return insts

def extract_insts_from_path(d: str):
    p_dir_insts = re.compile('/(\d+)/.*.txt')
    m = p_dir_insts.search(d)
    assert m is not None
    return int(m.group(1))


# this functions get all stats dumps
def get_all_chunks(stat_file: str, config_file: str, insts_from_dir):
    buff = []
    chunks = {}
    p_insts = re.compile('cpus?.committedInsts\s+(\d+)\s+#')
    if not insts_from_dir:
        insts = extract_insts_from_config(config_file)
    else:
        insts = extract_insts_from_path(stat_file)
    with open(expu(stat_file)) as f:
        for line in f:
            buff.append(line)
            if line.startswith('---------- End Simulation Statistics   ----------'):
                assert(insts)
                chunks[insts] = buff
                buff = []
            elif p_insts.search(line) is not None:
                insts += int(p_insts.search(line).group(1))
                insts = round(insts, -2)
    return chunks


def get_host_seconds(stat_file: str):
    p_host_time = re.compile('hostSeconds\s+(\S+)\s+#.*')
    host_time = 0.0
    with open(expu(stat_file)) as f:
        for line in f:
            m = p_host_time.match(line.strip())
            if m is not None:
                host_time += float(m.group(1))
            else:
                if line.startswith('hostSeconds'):
                    print(line)
                    assert False
    assert host_time > 0.1
    return host_time
    

# this functions get only the last dumps or the nearest stats dump
def get_raw_stats_around(stat_file: str, insts: int=200*(10**6),
                         cycles: int = None)-> list:
    buff = []

    # p_insts = re.compile('(?:cpus?|switch_cpus_1)\.committedInsts\s+(\d+)\s+#')
    # p_cycles = re.compile('(?:cpus?|switch_cpus_1)\.numCycles\s+(\d+)\s+#')

    print(stat_file)
    for line in reverse_readline(expu(stat_file)):
        buff.append(line)

        if line.startswith('---------- Begin Simulation Statistics ----------'):
            return buff

    return []
    # raise Exception("No beginning line found")


def to_num(x: str) -> (int, float):
    if '.' in x:
        return float(x)
    else:
        return int(x)

time_pattern = re.compile('\[PERF \]\[time=\s*(\d+)\].*')
def xs_get_time(line):
    return int(time_pattern.match(line).group(1))


def xs_get_mshr_latency(line: str, lv: str):
    # TOP.SimTop.l_soc.core_with_l2.l2cache.
    
    if lv == 'l2':
        pattern = re.compile(r"\[PERF \]\[time=\s+\d+\] TOP\.SimTop\.l_soc\.core_with_l2\.l2cache\.slices_(?P<bank>\d+)\.mshrCtl: mshr_latency_(?P<mshr_id>\d+)_(?P<lat_low>\d+)_(?P<lat_high>\d+),\s+(?P<count>\d+)")
    elif lv == 'l3':
        pattern = re.compile(r"\[PERF \]\[time=\s+\d+\] TOP\.SimTop\.l_soc\.l3cacheOpt\.slices_(?P<bank>\d+)\.mshrAlloc: mshr_latency_(?P<mshr_id>\d+)_(?P<lat_low>\d+)_(?P<lat_high>\d+),\s+(?P<count>\d+)")
    elif lv == 'l1d':
        pattern = re.compile(r"\[PERF \]\[time=\s+\d+\] TOP\.SimTop\.l_soc\.core_with_l2\.core\.memBlock\.dcache\.dcache\.missQueue\.entries_(?P<mshr_id>\d+): load_miss_penalty_to_use_(?P<lat_low>\d+)_(?P<lat_high>\d+),\s+(?P<count>\d+)")
    else:
        raise Exception("Unhandle cache levl")

    m = pattern.match(line)
    if m is None:
        return False, None, None, None, None
    mshr_id = int(m.group('mshr_id'))
    if 'bank' in m.groupdict(): 
        bank = int(m.group('bank'))
    else:
        bank = 0
    lat_low = int(m.group('lat_low'))
    lat_high = int(m.group('lat_high'))
    count = int(m.group('count'))
    lat_avg = (lat_high + lat_low) / 2.0
    total_latency = lat_avg * count

    if lv == 'l1d':
        print(f'{lv} mshr id: {mshr_id}, {lat_low}-{lat_high}: {total_latency} cycles')
    return True, int(bank), int(mshr_id), f'{lat_low}-{lat_high}', count



def xs_get_stats(stat_file: str, targets: list,
              insts: int=200*(10**6), re_targets=False) -> dict:
    
    target_keys = list(targets.keys())
    num_cores = len(list(filter(lambda x: x.startswith('commitInstr'), target_keys)))

    if not os.path.isfile(expu(stat_file)):
        print(stat_file)
    assert(os.path.isfile(expu(stat_file)))
    with open(stat_file) as f:
        lines = f.read().splitlines()

    if lines is None:
        return None

    patterns = {}
    accumulate_table = {}  # key: pattern, value: (count, [matched values])
    for k, p in targets.items():
        if isinstance(p, str):
            patterns[k] = re.compile(p)
        else:
            patterns[k] = re.compile(p[0])
            accumulate_table[k] = (p[1], [])
    stats = {}

    mshr_latency_l2 = {}
    mshr_latency_l3 = {}
    mshr_latency_l1d = {}
    total_access = 0
    total_match = 0
    keep_last = True
    # caches = ['l2', 'l3', 'l1d']
    caches = ['l2', 'l3']
    # caches = ['l1d']
    for ln, line in enumerate(lines):
        matched_re_pattern = False
        for k in patterns:
            m = patterns[k].search(line)
            if not m is None:
                matched_re_pattern = True
                if k in accumulate_table:
                    accumulate_table[k][1].append(to_num(m.group(1)))
                    print(accumulate_table)
                    # print(f"Matched {k} at line {ln}: {m.group(0)}")
                    # raise
                else:
                    print(k, patterns[k])
                    stats[k] = to_num(m.group(1))
                # print(f"Matched {k} at line {ln}: {m.group(0)}")
                # print(m.group(0))
                break
        if not matched_re_pattern:
            for lv in caches:
                mshr_latency = eval(f'mshr_latency_{lv}')
                matched, bank, mshr_id, bucket, count = xs_get_mshr_latency(line, lv)
                if matched:
                    total_match += 1
                    mshr_latency[bank] = mshr_latency.get(bank, {})
                    mshr_latency[bank][bucket] = mshr_latency[bank].get(bucket, {})
                    if mshr_id in mshr_latency[bank][bucket] and count != 0:
                        total_access += count
                    if keep_last:
                        mshr_latency[bank][bucket][mshr_id] = count
                    else:
                        if mshr_id not in mshr_latency[bank][bucket]:
                            mshr_latency[bank][bucket][mshr_id] = count
    print(accumulate_table)
    for k in accumulate_table:
        stats[k] = sum(accumulate_table[k][1][-accumulate_table[k][0]:])
    
    desired_keys = set(patterns.keys())
    obtained_keys = set(stats.keys())
    not_found_keys = desired_keys - obtained_keys
    print(not_found_keys)
    assert len(not_found_keys) == 0

    # print(mshr_latency.keys())
    tdf = None
    for lv in caches:
        bucket_sum = {}
        mshr_latency = eval(f'mshr_latency_{lv}')
        for bank in range(4):
            if bank not in mshr_latency:
                continue
            for bucket in mshr_latency[bank]:
                # print(mshr_latency[bank][bucket])
                s = sum(mshr_latency[bank][bucket].values())
                bucket_sum[bucket] = bucket_sum.get(bucket, 0) + s
        check_count = 0
        
        print(bucket_sum)
        if tdf is None:
            tdf = pd.DataFrame.from_dict(bucket_sum, orient='index', columns=[lv])
        else:
            tdf[lv] = pd.DataFrame.from_dict(bucket_sum, orient='index', columns=[lv])

        for k, v in bucket_sum.items():
            check_count += v
        print(f'total access: {total_access}, check count: {check_count}, total match: {total_match}')
    

    for lv in caches:
        # for i in tdf.index[:3]:
        #     tdf.at[i, lv] = 0
        tdf[f'{lv}_cum_sum'] = tdf[lv].cumsum()
        tdf[f'{lv}_cum_pct'] = 100*tdf[f'{lv}_cum_sum']/tdf[lv].sum()
        tdf.drop([f'{lv}_cum_sum'], axis=1, inplace=True)

    print(tdf)
    # print(stats)

    if not ('commitInstr' in stats and 'total_cycles' in stats):
        print("Warn: rob_commitInstr or totalCycle not exists")
        stats['ipc'] = 0
    else:
        stats['ipc'] = stats['commitInstr']/stats['total_cycles']
        for c in range(1, num_cores):
            stats[f'ipc{c}'] = stats[f'commitInstr{c}']/stats['total_cycles']
    return stats


def gem5_get_stats(stat_file: str, targets: list,
              insts: int=100*(10**6), re_targets=False,
              all_chunks=False, config_file=None, insts_from_dir=False) -> dict:
    if not os.path.isfile(expu(stat_file)):
        print(stat_file)
    assert(os.path.isfile(expu(stat_file)))

    patterns = {}

    if re_targets:
        meta_pattern = re.compile('.*\((\w.+)\).*')
        for t in targets:
            meta = meta_pattern.search(t).group(1)
            patterns[meta] = re.compile(t+'\s+(\d+\.?\d*)\s+')
    else:
        for t in targets:
            patterns[t] = re.compile(t+'\s+(\d+\.?\d*)\s+')

    if not all_chunks:
        lines = get_raw_stats_around(stat_file, insts)
        stats = {}

        # sim_time = get_host_seconds(stat_file)
        # stats['time'] = sim_time

        for line in lines:
            for k in patterns:
                m = patterns[k].search(line)
                if not m is None:
                    if re_targets:
                        # print(line)
                        # print(m.group(1))
                        # print(m.group(2))
                        stats[m.group(1)] = to_num(m.group(2))
                    else:
                        stats[k] = to_num(m.group(1))
        return stats
    else:
        assert config_file is not None
        chunks = get_all_chunks(stat_file, config_file, insts_from_dir)
        chunk_stats = {}
        for insts, chunk in chunks.items():
            chunk_stats[insts] = {}
            assert re_targets
            meta_pattern = re.compile('.*\((\w.+)\).*')

            for line in chunk:
                for k in patterns:
                    m = patterns[k].search(line)
                    if not m is None:
                        chunk_stats[insts][m.group(1)] = to_num(m.group(2))
        return chunk_stats


def get_stats(*args, **kwargs) -> dict:
    return gem5_get_stats(*args, **kwargs)


def get_stats_file_name(d: str):
    assert(os.path.isdir(d))
    results = []
    for f in os.listdir(d):
        if os.path.isfile(pjoin(d, f)) and f.endswith('stats.txt'):
            results.append(f)

    if len(results) > 1:
        print("Multiple stats file found!")
        assert False

    elif len(results) == 1:
        return results[0]

    else:
        return None


def get_stats_from_parent_dir(d: str, selected_benchmarks: [], *args, **kwargs):
    ret = {}
    assert(os.path.isdir(d))
    for sub_d in os.listdir(d):
        if os.path.isdir(pjoin(d, sub_d)):
            if '_' in sub_d:
                bmk = sub_d.split('_')[0]
            else:
                bmk = sub_d
            if selected_benchmarks is not None and bmk not in selected_benchmarks:
                continue

            stat_file = get_stats_file_name(pjoin(d, sub_d))
            if stat_file is not None:
                ret[sub_d] = get_stats(pjoin(d, sub_d, stat_file), *args, **kwargs)
    return ret


def add_st_ipc(hpt: str, d: dict, tid: int) -> None:
    num_insts = int(d['committedInsts'])
    if num_insts == 200*10**6:
        make_st_stat_cache()
        df = pd.read_csv(st_cache, index_col=0)
        d['st_ipc_{}'.format(tid)] = df.loc['ipc'][hpt]
    else:
        t = ['cpu\.(ipc)']
        st_d = get_stats(
            pjoin(st_stat_dir, hpt, 'stats.txt'), t,
            num_insts, re_targets=True
        )

        d['st_ipc_{}'.format(tid)] = st_d['ipc']

def add_ipc_pred(d: dict) -> None:
    real_ipc = float(d['st_ipc_0'])
    pred_ipc = float(d['HPTpredIPC'])
    d['IPC prediction error'] = (pred_ipc - real_ipc) / real_ipc

def add_slot_sanity(d: dict) -> None:
    d['slot sanity'] = \
            (float(d['numMissSlots::0']) + \
             float(d['numWaitSlots::0']) + \
             float(d['numBaseSlots::0'])) / (float(d['numCycles']) * 8)

def add_qos(tid: int, d: dict) -> None:
    solo_ipc = float(d['st_ipc_' + str(tid)])
    smt_ipc = float(d['ipc::' + str(tid)])
    d['QoS_' + str(tid)] = smt_ipc / solo_ipc

def add_overall_qos(hpt: str, lpt: str, d: dict) -> None:
    add_st_ipc(hpt, d, 0)
    add_st_ipc(lpt, d, 1)
    add_qos(0, d)
    add_qos(1, d)

    d['overall QoS'] = d['QoS_0'] + d['QoS_1']

def add_branch_mispred(d: dict) -> None:
    branches = float(d['branches'])
    mispred = float(d.get('branchMispredicts', 0.0))
    ind_mispred = float(d.get('indirectMispred', 0.0))
    d['mispredict rate'] = mispred / branches
    print('Commit instr', d['Insts'], mispred)
    d['total branch MPKI'] = mispred / float(d['Insts']) * 1000
    d['indirect branch MPKI'] = ind_mispred / float(d['Insts']) * 1000
    d['direct branch MPKI'] = d['total branch MPKI'] - d['indirect branch MPKI']
    d['return MPKI'] = float(d['RASIncorrect']) / float(d['Insts']) * 1000

def add_mem_bw(d: dict) -> None:
    to_mc_total = float(d.get('WritebackDirty', 0)) + float(d.get('ReadResp', 0)) + float(d.get('ReadExResp', 0))
    d['BW_MB/s'] = to_mc_total * 64 / float(d['Sec']) / 1024 / 1024

def xs_add_branch_mispred(d: dict) -> None:
    mispred = float(d['BpBWrong']) + float(d['BpJWrong']) + float(d['BpIWrong'])
    branches = float(d['BpInstr'])
    d['mispredict rate'] = mispred / branches
    d['total branch MPKI'] = mispred / float(d['commitInstr']) * 1000
    # d['indirect branch MPKI'] = ind_mispred / float(d['Insts']) * 1000
    # d['direct branch MPKI'] = d['total branch MPKI'] - d['indirect branch MPKI']
    # d['return MPKI'] = float(d['RASIncorrect']) / float(d['Insts']) * 1000

def add_cache_mpki(d: dict) -> None:
    # d['L2/L1 acc'] = float(d['l2.overallAccesses']) / float(d['dcache.overallAccesses'])
    d['L3.NonPrefAcc'] = float(d.get('l3.demandAcc', 0.0) - d.get('l3.demandAccesses::l2.pref', 0.0))
    d['L3.NonPrefMiss'] = float(d.get('l3.demandMiss', 0.0) - d.get('l3.demandMisses::l2.pref', 0.0))
    d.pop('l3.demandAccesses::l2.pref', None)
    d.pop('l3.demandMisses::l2.pref', None)

    d['L2.MPKI'] = float(d.get('l2.demandMiss', 0.0)) / float(d['Insts']) * 1000
    d['L3.NonPref.MPKI'] = float(d.get('L3.NonPrefMiss', 0.0)) / float(d['Insts']) * 1000

    d['l1D.MPKI'] = float(d.get('dcache.demandMiss', 0.0)) / float(d['Insts']) * 1000

    # if 'icache.demandMisses' in d:
    #     d['L1I_MPKI'] = float(d['icache.demandMisses']) / float(d['Insts']) * 1000
    # else:
    #     d['L1I_MPKI'] = 0.0

def xs_add_cache_mpki(d: dict) -> None:
    # L2/L3
    n_banks = 4
    if 'l2_acc' not in d:
        print('Using latest xs')
        print(d)
        for cache in ['l2', 'l3']:
            d[f'{cache}_acc'] = 0
            d[f'{cache}_hit'] = 0
            d[f'{cache}_recv_pref'] = 0
            for i in range(n_banks):  # TODO remove hardcode
                d[f'{cache}_acc'] += d[f'{cache}b{i}_acc']
                d.pop(f'{cache}b{i}_acc')
                d[f'{cache}_hit'] += d[f'{cache}b{i}_hit']
                d.pop(f'{cache}b{i}_hit')
                d[f'{cache}_recv_pref'] += d[f'{cache}b{i}_recv_pref']
                d.pop(f'{cache}b{i}_recv_pref')
    else:
        print('Using old xs')
    for cache in ['l2', 'l3']:
        d[f'{cache.upper()}.overallMiss'] = d[f'{cache}_acc'] - d[f'{cache}_hit']
        # d[f'{cache.upper()}_overall_miss_rate'] = d[f'{cache.upper()}_overall_miss'] / (d[f'{cache}_acc'] + 1)
        d[f'{cache.upper()}.overallMPKI'] = d[f'{cache.upper()}.overallMiss'] / (d[f'{cache}_acc'] + 1)
        d.pop(f'{cache}_hit')
    for cache in ['l2']:
        d[f'{cache}.NonPrefAcc'] = d[f'{cache}_acc'] - d[f'{cache}_recv_pref']
        # d[f'{cache}_demand_mpki'] = d[f'{cache}_demand_acc'] / d['commitInstr'] * 1000

    # L1D
    d[f'L1D.miss'] = 0
    d[f'L1D.acc'] = 0
    for load_pipeline in range(2):  # TODO remove hardcode
        d[f'L1D.miss'] += d[f'l1d_{load_pipeline}_miss']
        d.pop(f'l1d_{load_pipeline}_miss')
        d[f'L1D.acc'] += d[f'l1d_{load_pipeline}_acc']
        d.pop(f'l1d_{load_pipeline}_acc')
    d[f'L1D.MPKI'] = d[f'L1D.miss'] / d['commitInstr'] * 1000

# d['layer3_integet_dq'] = d['stall_cycle_int_dq'] / d['stall_cycles_core']
# d['layer3_floatpoint_dq'] = d['stall_cycle_fp_dq'] / d['stall_cycles_core']
# d['layer3_rob'] = d['stall_cycle_rob'] / d['stall_cycles_core']
# d['layer3_integet_prf'] = d['stall_cycle_int'] / d['stall_cycles_core']
# d['layer3_floatpoint_prf'] = d['stall_cycle_fp'] / d['stall_cycles_core']
# d['layer3_lsu_ports'] = d['ls_dq_bound_cycles'] / d['stall_cycles_core']

topdown_filter = [
    'layer1_frontend_bound',
    'layer1_bad_speculation',
    'layer1_retiring',
    # 'layer2_memory_bound',
    'layer2_core_bound',
    # 'layer3_integet_dq',
    # 'layer3_floatpoint_dq',
    # 'layer3_rob',
    # 'layer3_integet_prf',
    # 'layer3_floatpoint_prf',
    # 'layer3_lsu_ports',
    'layer2_memory_bound-tlb',
    'layer2_memory_bound-non_tlb',
    # 'DTlbStall',
    'ipc',
]

def topdown_post_process(d: dict) -> None:
    print(d)
    d['total_slots'] = d['Cycles'] * 6
    d['layer1_frontend_bound'] = (d['IcacheStall'] + d['ITlbStall'] + d['FragStall'] + d['FetchBufferInvalid'] +
                                  d['SerializeStall'] + d['OtherFetchStall']) / d['total_slots']
    d['layer1_bad_speculation'] = (d['BpStall'] + d['SquashStall'] + d['InstMisPred'] +
                                   d['InstSquashed'] + d['CommitSquash']) / d['total_slots']
    d['layer1_retiring'] = d['NoStall'] / d['total_slots']

    d['layer2_memory_bound-tlb'] = d['DTlbStall'] / d['total_slots']
    d['layer2_memory_bound-non_tlb'] = (d['LoadL1Bound'] + d['LoadL2Bound'] + d['LoadL3Bound'] +
                                d['LoadMemBound'] + d['StoreL1Bound'] + d['StoreL2Bound'] + d['StoreL3Bound'] +
                                d['StoreMemBound']) / d['total_slots']
    d['layer2_memory_bound'] = (d['DTlbStall'] + d['LoadL1Bound'] + d['LoadL2Bound'] + d['LoadL3Bound'] +
                                d['LoadMemBound'] + d['StoreL1Bound'] + d['StoreL2Bound'] + d['StoreL3Bound'] +
                                d['StoreMemBound']) / d['total_slots']
    d['layer2_core_bound'] = (d['LongExecute'] + d['InstNotReady'] + d['OtherStall'] + d['ResumeUnblock']
                                 ) / d['total_slots']
    d['layer1_backend_bound'] = d['layer2_memory_bound'] + d['layer2_core_bound']
    # for k in t.LievenStalls:
    #     d.pop(k)
    keys = list(d.keys())
    for k in keys:
        if k not in topdown_filter:
            d.pop(k)


def xs_topdown_post_process(d: dict) -> None:
    d['total_slots'] = d['total_cycles'] * 6
#     # top = layer 1
#     frontend_bound = top.add_down("Frontend Bound", use('decode_bubbles') / use('total_slots'))
#     bad_speculation = top.add_down("Bad Speculation", (use('slots_issued') - use('slots_retired') + use('recovery_bubbles')) / use('total_slots'))
#     retiring = top.add_down("Retiring", use('slots_retired') / use('total_slots'))
#     backend_bound = top.add_down("Backend Bound", top - frontend_bound - bad_speculation - retiring)
    d['layer1_frontend_bound'] = d['decode_bubbles'] / d['total_slots']
    d['layer1_bad_speculation'] = (d['slots_issued'] - d['slots_retired'] + d['recovery_bubbles']) / d['total_slots']
    d['layer1_retiring'] = d['slots_retired'] / d['total_slots']
    d['layer1_backend_bound'] = 1 - d['layer1_frontend_bound'] - d['layer1_bad_speculation'] - d['layer1_retiring']

# #top->frontend_bound
#     fetch_latency = frontend_bound.add_down("Fetch Latency", use('fetch_bubbles') / use('total_slots'))
#     fetch_bandwidth = frontend_bound.add_down("Fetch Bandwidth", frontend_bound - fetch_latency)
    d['layer2_fetch_latency'] = d['fetch_bubbles'] / d['total_slots']
    d['layer2_fetch_bandwidth'] = d['layer1_frontend_bound'] - d['layer2_fetch_latency']

# # top->frontend_bound->fetch_latency
#     itlb_miss = fetch_latency.add_down("iTLB Miss", use('itlb_miss_cycles') / use('total_cycles'))
#     icache_miss = fetch_latency.add_down("iCache Miss", use('icache_miss_cycles') / use('total_cycles'))
#     stage2_redirect_cycles = fetch_latency.add_down("Stage2 Redirect", use('stage2_redirect_cycles') / use('total_cycles'))
#     if2id_bandwidth = fetch_latency.add_down("IF2ID Bandwidth", use('ifu2id_hvButNotFull_slots') / use('total_slots'))
#     fetch_latency_others = fetch_latency.add_down("Fetch Latency Others", fetch_latency - itlb_miss - icache_miss - stage2_redirect_cycles - if2id_bandwidth)
#     csv_file['ifu2id_allNO_slots'] = use('ifu2id_allNO_cycle') * 6
# csv_file['ifu2id_hvButNotFull_slots'] = use('fetch_bubbles') - use('ifu2id_allNO_slots')
    d['ifu2id_allNO_slots'] = d['ifu2id_allNO_cycle'] * 6
    d['ifu2id_hvButNotFull_slots'] = d['fetch_bubbles'] - d['ifu2id_allNO_slots']
    d['layer3_itlb_miss'] = d['itlb_miss_cycles'] / d['total_cycles']
    d['layer3_icache_miss'] = d['icache_miss_cycles'] / d['total_cycles']
    d['layer3_stage2_redirect_cycles'] = d['stage2_redirect_cycles'] / d['total_cycles']
    d['layer3_if2id_bandwidth'] = d['ifu2id_hvButNotFull_slots'] / d['total_slots']
    d['layer3_fetch_latency_others'] = d['layer2_fetch_latency'] - d['layer3_itlb_miss'] - d['layer3_icache_miss'] - d['layer3_stage2_redirect_cycles'] - d['layer3_if2id_bandwidth']

# # top->frontend_bound->fetch_latency->stage2_redirect_cycles
#     branch_resteers = stage2_redirect_cycles.add_down("Branch Resteers", use('branch_resteers_cycles') / use('total_cycles'))
#     robFlush_bubble = stage2_redirect_cycles.add_down("RobFlush Bubble", use('robFlush_bubble_cycles') / use('total_cycles'))
#     ldReplay_bubble = stage2_redirect_cycles.add_down("LdReplay Bubble", use('ldReplay_bubble_cycles') / use('total_cycles'))
    d['layer4_branch_resteers'] = d['branch_resteers_cycles'] / d['total_cycles']
    d['layer4_robFlush_bubble'] = d['robFlush_bubble_cycles'] / d['total_cycles']
    d['layer4_ldReplay_bubble'] = d['ldReplay_bubble_cycles'] / d['total_cycles']

# # top->bad_speculation
#     branch_mispredicts = bad_speculation.add_down("Branch Mispredicts", bad_speculation)

# # top->backend_bound
    # stall_cycles_core = use('stall_cycle_fp') + use('stall_cycle_int') + use('stall_cycle_rob') + use('stall_cycle_int_dq') + use('stall_cycle_fp_dq') + use('ls_dq_bound_cycles')
#     memory_bound = backend_bound.add_down("Memory Bound", backend_bound * (use('store_bound_cycles') + use('load_bound_cycles')) / (
#         stall_cycles_core + use('store_bound_cycles') + use('load_bound_cycles')))
#     core_bound = backend_bound.add_down("Core Bound", backend_bound - memory_bound)
    d['stall_cycles_core'] = d['stall_cycle_fp'] + d['stall_cycle_int'] + d['stall_cycle_rob'] + \
        d['stall_cycle_int_dq'] + d['stall_cycle_fp_dq'] + d['ls_dq_bound_cycles']

    d['layer2_memory_bound'] = d['layer1_backend_bound'] * (d['store_bound_cycles'] + d['load_bound_cycles']) / (
        d['stall_cycles_core'] + d['store_bound_cycles'] + d['load_bound_cycles'])

    d['layer2_core_bound'] = d['layer1_backend_bound'] - d['layer2_memory_bound']

# # top->backend_bound->core_bound
#     integer_dq = core_bound.add_down("Integer DQ", core_bound * use('stall_cycle_int_dq') / stall_cycles_core)
#     floatpoint_dq = core_bound.add_down("Floatpoint DQ", core_bound * use('stall_cycle_fp_dq') / stall_cycles_core)
#     rob = core_bound.add_down("ROB", core_bound * use('stall_cycle_rob') / stall_cycles_core)
#     integer_prf = core_bound.add_down("Integer PRF", core_bound * use('stall_cycle_int') / stall_cycles_core)
#     floatpoint_prf = core_bound.add_down("Floatpoint PRF", core_bound * use('stall_cycle_fp') / stall_cycles_core)
#     lsu_ports = core_bound.add_down("LSU Ports", core_bound * use('ls_dq_bound_cycles') / stall_cycles_core)
    d['layer3_integet_dq'] = d['stall_cycle_int_dq'] / d['stall_cycles_core']
    d['layer3_floatpoint_dq'] = d['stall_cycle_fp_dq'] / d['stall_cycles_core']
    d['layer3_rob'] = d['stall_cycle_rob'] / d['stall_cycles_core']
    d['layer3_integet_prf'] = d['stall_cycle_int'] / d['stall_cycles_core']
    d['layer3_floatpoint_prf'] = d['stall_cycle_fp'] / d['stall_cycles_core']
    d['layer3_lsu_ports'] = d['ls_dq_bound_cycles'] / d['stall_cycles_core']

# # top->backend_bound->memory_bound
#     stores_bound = memory_bound.add_down("Stores Bound", use('store_bound_cycles') / use('total_cycles'))
#     loads_bound = memory_bound.add_down("Loads Bound", use('load_bound_cycles') / use('total_cycles'))
    d['layer2_stores_bound'] = d['store_bound_cycles'] / d['total_cycles']
    d['layer2_loads_bound'] = d['load_bound_cycles'] / d['total_cycles']

# # top->backend_bound->memory_bound->loads_bound
#     l1d_loads_bound = loads_bound.add_down("L1D Loads", use('l1d_loads_bound_cycles') / use('total_cycles'))
#     l2_loads_bound = loads_bound.add_down("L2 Loads", use('l2_loads_bound_cycles') / use('total_cycles'))
#     l3_loads_bound = loads_bound.add_down("L3 Loads", use('l3_loads_bound_cycles') / use('total_cycles'))
#     ddr_loads_bound = loads_bound.add_down("DDR Loads", use('ddr_loads_bound_cycles') / use('total_cycles'))
    d['layer3_l1d_loads_bound'] = d['l1d_loads_bound_cycles'] / d['total_cycles']
    d['layer3_l2_loads_bound'] = d['l2_loads_bound_cycles'] / d['total_cycles']
    d['layer3_l3_loads_bound'] = d['l3_loads_bound_cycles'] / d['total_cycles']
    d['layer3_ddr_loads_bound'] = d['ddr_loads_bound_cycles'] / d['total_cycles']

# # top->backend_bound->memory_bound->loads_bound->l1d_loads_bound
#     l1d_loads_mshr_bound = l1d_loads_bound.add_down("L1D Loads MSHR", use('l1d_loads_mshr_bound') / use('total_cycles'))
#     l1d_loads_tlb_bound = l1d_loads_bound.add_down("L1D Loads TLB", use('l1d_loads_tlb_bound') / use('total_cycles'))
#     l1d_loads_store_data_bound = l1d_loads_bound.add_down("L1D Loads sdata", use('l1d_loads_store_data_bound') / use('total_cycles'))
#     l1d_loads_bank_conflict_bound = l1d_loads_bound.add_down("L1D Loads\nBank Conflict", use('l1d_loads_bank_conflict_bound') / use('total_cycles'))
#     l1d_loads_vio_check_redo_bound = l1d_loads_bound.add_down("L1D Loads VioRedo", use('l1d_loads_vio_check_redo_bound') / use('total_cycles'))
    d['layer4_l1d_loads_mshr_bound'] = d['l1d_loads_mshr_bound'] / d['total_cycles']
    d['layer4_l1d_loads_tlb_bound'] = d['l1d_loads_tlb_bound'] / d['total_cycles']
    d['layer4_l1d_loads_store_data_bound'] = d['l1d_loads_store_data_bound'] / d['total_cycles']
    d['layer4_l1d_loads_bank_conflict_bound'] = d['l1d_loads_bank_conflict_bound'] / d['total_cycles']
    d['layer4_l1d_loads_vio_check_redo_bound'] = d['l1d_loads_vio_check_redo_bound'] / d['total_cycles']

    keys = list(d.keys())
    for k in keys:
        if k not in topdown_filter:
            d.pop(k)

def add_warmup_mpki(d: dict) -> None:
    d['L2MPKI'] = float(d.get('l2.demandMisses', 0)) / float(d['Insts']) * 1000
    d['L3MPKI'] = float(d.get('l3.demandMisses', 0)) / float(d['Insts']) * 1000
    if 'branchMispredicts' in d:
        mispred = float(d['branchMispredicts'])
    else:
        mispred = 0.0
    d['total branch MPKI'] = mispred / float(d['Insts']) * 1000

def add_fanout(d: dict) -> None:
    large_fanout = float(d.get('largeFanoutInsts', 0)) + 1.0
    d['LF_rate'] = large_fanout / float(d.get('Insts', 200 * 10**6))
    # print(large_fanout)
    d['FP_rate'] = float(d.get('falsePositiveLF', 0)) / large_fanout
    d['FN_rate'] = float(d.get('falseNegativeLF', 0)) / large_fanout
    del d['falsePositiveLF']
    del d['falseNegativeLF']


def get_spec2017_int():
    with open(os.path.expanduser(env.data('int.txt'))) as f:
        return [x for x in f.read().split('\n') if len(x) > 1]


def get_spec2017_fp():
    with open(os.path.expanduser(env.data('fp.txt'))) as f:
        return [x for x in f.read().split('\n') if len(x) > 1]


def get_all_spec2017_simpoints():
    benchmarks = get_spec2017_int() + get_spec2017_fp()
    points = []
    for b in benchmarks:
        for i in range(0, 3):
            points.append(f'{b}_{i}')
    return points

def add_packet(d: dict) -> None:
    d['by_bw'] = d['Insts'] / (d['TotalP']/3.1)
    d['by_chasing'] = d['Insts'] / (d['TotalP']/4.0)
    d['by_crit_ptr'] = min(d['Insts'] / (d['KeySrcP']/4), 4.0, d['TotalP']/10.0)


def find_stats_file(d: str) -> str:
    assert osp.isdir(d)
    stats = []
    for file in os.listdir(d):
        if file.endswith('stats.txt') and osp.isfile(pjoin(d, file)):
            stats.append(pjoin(d, file))
    assert len(stats) == 1
    return stats[0]


def get_df(path_dict: dict, arch: str, targets, filter_bmk=None):
    path = path_dict[arch]
    matrix = {}
    print(path)
    for d in os.listdir(path):
        if filter_bmk is not None and not d.startswith(filter_bmk):
            continue
        stat_dir_path = osp.join(path, d)
        if not osp.isdir(stat_dir_path):
            continue
        f = find_stats_file(stat_dir_path)
        tup = get_stats(f, targets, re_targets=True)
        gatrix[d] = tup
    df = pd.DataFrame.from_dict(matrix, orient='index')
    return df


def scale_tick(df: pd.DataFrame):
    for col in df:
        if 'Tick' in col:
            df[col] = df[col] / 500.0
    return df


def get_spec_ref_time(bmk, ver):
    if ver == '06':
        top = "/nfs-nvme/home/share/cpu2006v99/benchspec/CPU2006"
        ref_file = "/nfs-nvme/home/share/cpu2006v99/benchspec/CPU2006/{}/data/ref/reftime"
    else:
        assert ver == '17'
        top = "/home/zyy/research-data/spec2017_20201126/benchspec/CPU"
        ref_file = "/home/zyy/research-data/spec2017_20201126/benchspec/CPU/{}/data/refrate/reftime"

    codename = None
    for d in os.listdir(top):
        if osp.isdir(osp.join(top, d)) and bmk in d and '_s' not in d:
            codename = d
    ref_file = ref_file.format(codename)

    pattern = re.compile(r'\d+')
    print(ref_file)
    assert osp.isfile(ref_file)
    with open(ref_file) as f:
        for line in f:
            m = pattern.search(line)
            if m is not None:
                return float(m.group(0))
    return None

if __name__ == '__main__':
    chunks = get_all_chunks(
            '/home51/zyy/expri_results/shotgun/gem5_shotgun_cont_06/FullWindowO3Config/gcc_200/0/m5out/stats.txt',
            '/home51/zyy/expri_results/shotgun/gem5_shotgun_cont_06/FullWindowO3Config/gcc_200/0/m5out/config.json',
            )
    print(chunks.keys())
