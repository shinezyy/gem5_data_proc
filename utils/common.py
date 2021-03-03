import sys
from os.path import join as pjoin
from os.path import expanduser as expu
import re
import json
from copy import deepcopy
import pandas as pd

from local_configs import Env
from paths import *

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


# this functions get only the last dumps or the nearest stats dump
def get_raw_stats_around(stat_file: str, insts: int=200*(10**6),
                         cycles: int = None)-> list:
    old_buff = []
    buff = []

    old_insts = None
    old_cycles = None
    new_insts = None
    new_cycles = None

    p_insts = re.compile('cpus?.committedInsts\s+(\d+)\s+#')
    p_cycles = re.compile('cpus?.numCycles\s+(\d+)\s+#')

    if insts > 500*(10**6):
        for line in reverse_readline(expu(stat_file)):
            buff.append(line)

            if line.startswith('---------- Begin Simulation Statistics ----------'):
                if cycles:
                    pass
                else:
                    if insts >= new_insts:
                        if old_insts is None:
                            return buff
                        elif abs(insts - old_insts) > abs(new_insts - insts):
                            return buff
                        else:
                            return old_buff
                    else:
                        old_insts = new_insts
                        old_cycles = new_cylces
                        old_buff = deepcopy(buff)
                        buff.clear()

            elif line.startswith('system.switch_cpus.committedInsts'):
                new_insts = int(p_insts.search(line).group(1))
            elif line.startswith('system.switch_cpus.numCycles'):
                new_cylces = int(p_cycles.search(line).group(1))

    else:
        with open(expu(stat_file)) as f:
            for line in f:
                buff.append(line)

                if line.startswith('---------- End Simulation Statistics   ----------'):
                    if cycles:
                        pass
                    else:
                        if insts <= new_insts:
                            if old_insts is None:
                                return buff
                            elif abs(insts - old_insts) > abs(new_insts - insts):
                                return buff
                            else:
                                return old_buff
                        else:
                            old_insts = new_insts
                            old_cycles = new_cylces
                            old_buff = deepcopy(buff)
                            buff.clear()

                elif p_insts.search(line) is not None:
                    new_insts = int(p_insts.search(line).group(1))
                elif p_cycles.search(line) is not None:
                    new_cylces = int(p_cycles.search(line).group(1))

    return old_buff


def to_num(x: str) -> (int, float):
    if '.' in x:
        return float(x)
    else:
        return int(x)

def xs_get_time(line):
    time_parrern = re.compile('.*\[time=\s*(\d+)\].*')
    return int(time_parrern.search(line).group(1))


def xs_get_raw_stats_around(stat_file: str)-> list:

    buff = []
    time = 0

    for line in reverse_readline(expu(stat_file)):
        if line.startswith('[PERF ]'):
            if time == 0:
                time = xs_get_time(line)
                # print(time)

            if time != 0 and xs_get_time(line) != time:
                buff.append('totalCycle,' + str(time - xs_get_time(line)))
                return buff

            buff.append(line)

    return None

def xs_get_stats(stat_file: str, targets: list,
              insts: int=200*(10**6), re_targets=False) -> dict:
    if not os.path.isfile(expu(stat_file)):
        print(stat_file)
    assert(os.path.isfile(expu(stat_file)))
    lines = xs_get_raw_stats_around(stat_file)

    if lines is None:
        return None

    patterns = {}
    if re_targets:
        meta_pattern = re.compile('.*\((\w.+)\).*')
        for t in targets:
            meta = meta_pattern.search(t).group(1)
            # patterns[meta] = re.compile(t+'\s+(\d+\.?\d*)\s+')
            patterns[meta] = re.compile('.*?' + meta + ',\s*(\d+)')
    else:
        for t in targets:
            # patterns[t] = re.compile(t+'\s+(\d+\.?\d*)\s+')
            patterns[t] = re.compile('.*?' + meta + ',\s*(\d+)')

    # print(patterns)
    stats = {}

    for line in lines:
        for k in patterns:
            m = patterns[k].search(line)
            if not m is None:
                if re_targets:
                    stats[k] = to_num(m.group(1))
                else:
                    stats[k] = to_num(m.group(1))
    if not ('roq_commitInstr' in stats and 'totalCycle' in stats):
        print("Warn: roq_commitInstr or totalCycle not exists")
        stats['ipc'] = 0
    else:
        stats['ipc'] = stats['roq_commitInstr']/stats['totalCycle']
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

        for line in lines:
            for k in patterns:
                m = patterns[k].search(line)
                if not m is None:
                    if re_targets:
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
    mispred = float(d['branchMispredicts'])
    d['mispredict rate'] = mispred / branches;
    d['MPKI'] = mispred / float(d['Insts']) * 1000;

def add_cache_mpki(d: dict) -> None:
    d['L2MPKI'] = float(d['l2.demand_misses']) / float(d['Insts']) * 1000;
    d['L3MPKI'] = float(d['l3.demand_misses']) / float(d['Insts']) * 1000;

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
        top = "/home/zyy/research-data/spec2006/benchspec/CPU2006"
        ref_file = "/home/zyy/research-data/spec2006/benchspec/CPU2006/{}/data/ref/reftime"
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
