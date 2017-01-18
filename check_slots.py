import argparse
import sh

parser = argparse.ArgumentParser(description='check sum of slots')

parser.add_argument('stats', metavar='stats',
                    help='stats.txt to be converted')

args = parser.parse_args()

# cmd = 'tail ' + args.stats + ' -n 100 | grep "QoS\|committedInsts::\|Slots::0\|ipc::\|HPT\|numCycles\|Overlapped"'

x = sh.grep(sh.grep("system.cpu.committedInsts::0 *200[0-9]\{6\}",
                    args.stats, '-m', "1", '-A', '50', '-B', '50'),
            "QoS\|committedInsts::\|Slots::0\|ipc::\|HPT\|numCycles\|Overlapped")

print x

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
    elif line.startswith('system.cpu.ipc::0'):
        d['system.cpu.ipc::0'] = line.split()[1]

base = float(d['system.cpu.fmt.numBaseSlots::0'])
wait = float(d['system.cpu.fmt.numWaitSlots::0'])
miss = float(d['system.cpu.fmt.numMissSlots::0'])
cycle = float(d['system.cpu.numCycles'])
ipc = float(d['system.cpu.ipc::0'])

print (base + wait + miss)/ (8*cycle)

qos = (base + miss) / (base + miss + wait)

print "Predicted QoS:", qos

print 'Predicted ST IPC:', ipc/qos
