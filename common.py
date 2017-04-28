import sys
import os
from os.path import join as pjoin
from os.path import expanduser as expu
import re
from copy import deepcopy


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
                if buffer[-1] is not '\n':
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
        return [x for x in pair_dirs if os.path.isdir(expu(x)) and '_' in x]
    else:
        return [x[1] for x in zip(pair_dirs, pairs_s) \
                if os.path.isdir(expu(x[0]))]

def stat_filt(dirs):
    # type: (list) -> list
    return [f for f in dirs if os.path.isfile(expu(pjoin(f, 'stats.txt')))]

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


def print_line():
    print('---------------------------------------------------------------')


def get_stats_around(stat_file: str, insts: int=200*(10**6), cycles: int = None)-> list:
    old_buff = []
    buff = []

    old_insts = None
    old_cycles = None
    new_insts = None
    new_cycles = None

    p_insts = re.compile('cpu.committedInsts::0\s+(\d+)\s+#')
    p_cycles = re.compile('cpu.numCycles\s+(\d+)\s+#')

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

        elif line.startswith('system.cpu.committedInsts::0'):
            new_insts = int(p_insts.search(line).group(1))
        elif line.startswith('system.cpu.numCycles'):
            new_cylces = int(p_cycles.search(line).group(1))

    return old_buff



def get_stats(stat_file: str, targets: list,
              insts: int=200*(10**6), re_targets=False) -> dict:
    assert(os.path.isfile(expu(stat_file)))
    lines = get_stats_around(stat_file, insts)

    patterns = {}
    if re_targets:
        meta_pattern = re.compile('.*\((.+)\).*')
        for t in targets:
            meta = meta_pattern.search(t).group(1)
            patterns[meta] = re.compile(t+'\s+(\d+\.?\d*)\s+#')
    else:
        for t in targets:
            patterns[t] = re.compile(t+'\s+(\d+\.?\d*)\s+#')

    # print(patterns)
    stats = {}

    for line in lines:
        for k in patterns:
            m = patterns[k].search(line)
            if not m is None:
                if re_targets:
                    stats[m.group(1)] = m.group(2)
                else:
                    stats[k] = m.group(1)
    return stats


