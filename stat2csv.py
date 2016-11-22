import argparse

parser = argparse.ArgumentParser(description='convert stats.txt to csv')

parser.add_argument('stats', metavar='stats',
                    help='stats.txt to be converted')
parser.add_argument('csv', metavar='csv',
                    help='csv to generate')
parser.add_argument('use', metavar='use',
                    help='file to indicate which values should be added to csv')

args = parser.parse_args()

attrs = dict()
with open(args.use) as inf:
    for line in inf:
        k = line.split()[0]
        attrs[k] = 0

print attrs

idx = 0
attrs_list = []
num_sim = -1
with open(args.stats) as inf, open(args.csv, 'w') as outf:
    for x in sorted(attrs):
        print >> outf, x, ',',
    print >>outf, 'local_ipc_0, local_ipc_1'
    for line in inf:
        x = line.split()
        if not len(x):
            continue
        if not line.startswith('---------- B'):
            k = x[0]
            if k in attrs:
                attrs[k] = x[1]
        else:
            num_sim += 1
            if num_sim == 0:
                continue
            elif num_sim > 0 and num_sim < 4:
                attrs_list.append(attrs)
                attrs = attrs.fromkeys(attrs, 0)
                print len(attrs_list)
            else:
                for x in sorted(attrs):
                    print >>outf, attrs[x], ',',
                local_ipc_0 = (float(attrs['system.cpu.committedInsts::0'])
                               - float(attrs_list[2]
                                       ['system.cpu.committedInsts::0']))/ \
                        (float(attrs['system.cpu.numCycles'])
                         - float(attrs_list[2]['system.cpu.numCycles']))
                print >>outf, local_ipc_0, ',',

                local_ipc_1 = (float(attrs['system.cpu.committedInsts::1'])
                               - float(attrs_list[2]
                                       ['system.cpu.committedInsts::1']))/ \
                        (float(attrs['system.cpu.numCycles'])
                         - float(attrs_list[2]['system.cpu.numCycles']))
                print >>outf, local_ipc_1

                attrs_list[0] = attrs_list[1]
                attrs_list[1] = attrs_list[2]
                attrs_list[2] = attrs
                # reset
                attrs = attrs.fromkeys(attrs, 0)


