import argparse
import sh

parser = argparse.ArgumentParser(description='check sum of slots')

parser.add_argument('stats', metavar='stats',
                    help='stats.txt to be converted')

args = parser.parse_args()

# cmd = 'tail ' + args.stats + ' -n 100 | grep "QoS\|committedInsts::\|Slots::0\|ipc::\|HPT\|numCycles\|Overlapped"'

x = sh.grep(sh.tail(args.stats, '-n', "100"),
            "QoS\|committedInsts::\|Slots::0\|ipc::\|HPT\|numCycles\|Overlapped")

d = dict()

for line in x:
    line = str(line)
    if line.startswith('system.cpu.numCycles'):
        d['system.cpu.numCycles'] = line.split()[1]
    elif line.startswith('system.cpu.fmt.numBaseSlots::0'):
        d['system.cpu.fmt.numBaseSlots::0'] = line.split()[1]
    elif line.startswith('system.cpu.fmt.numWaitSlots::0'):
        d['system.cpu.fmt.numWaitSlots::0'] = line.split()[1]
    elif line.startswith('system.cpu.fmt.numMissSlots::0'):
        d['system.cpu.fmt.numMissSlots::0'] = line.split()[1]

base = float(d['system.cpu.fmt.numBaseSlots::0'])
wait = float(d['system.cpu.fmt.numWaitSlots::0'])
miss = float(d['system.cpu.fmt.numMissSlots::0'])
cycle = float(d['system.cpu.numCycles'])

print (base + wait + miss)/ (8*cycle)
