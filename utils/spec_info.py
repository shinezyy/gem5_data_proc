import re
import os.path as osp

spec_bmks = {
        '06': {
            'int': [
                'perlbench',
                'bzip2',
                'gcc',
                'mcf',
                'gobmk',
                'hmmer',
                'sjeng',
                'libquantum',
                'h264ref',
                'omnetpp',
                'astar',
                'xalancbmk',
                ],
            'float':[
                ]
            },
        '17': {},
        }

def get_insts(fname: str):
    print(fname)
    assert osp.isfile(fname)
    p = re.compile('total guest instructions = (\d+)')
    with open(fname) as f:
        for line in f:
            m = p.search(line)
            if m is not None:
                return m.group(1)
    return None
