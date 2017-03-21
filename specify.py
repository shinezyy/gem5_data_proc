import sh
import re


def preproc_pattern(p):
    p += ' +(\d+\.?\d*)'
    return re.compile(p)


def specify_stat(stat_file, use_tail, stat_name):

    p = preproc_pattern(stat_name)

    # Get lines @ specified number of instructions
    if use_tail:
        raw_str = sh.tail(stat_file, '-n', '2000')
    else:
        raw_str = sh.grep("system.cpu.committedInsts::0 *2[0-9]\{8\}",
                        stat_file, '-m', "1", '-A', '1000', '-B', '600')

    # convert lines to dict for further use
    for line in raw_str:
        line = str(line)
        m = p.search(line)
        if not m is None:
            return m.group(1)
