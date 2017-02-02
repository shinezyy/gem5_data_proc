import argparse
import sh

parser = argparse.ArgumentParser(description='check sum of slots')

parser.add_argument('stats', metavar='stats',
                    help='stats.txt to be converted')

parser.add_argument('tail', metavar='tail',
                    help='use statistics in tail of stats.txt')

args = parser.parse_args()

# cmd = 'tail ' + args.stats + ' -n 100 | grep "QoS\|committedInsts::\|Slots::0\|ipc::\|HPT\|numCycles\|Overlapped"'

targets = [
    'committedInsts::0',
    'Slots::0',
    'ipc::',
    'HPT',
    'numCycles',
    'Overlapped',
    'rectified',
    'wait_to_miss::0',
    'system.cpu.iew.exec_branches::0',
    'system.cpu.iew.branchMispredicts::0',
]

tstr = 'Qos'
for s in targets:
    tstr += '\|'+s

if args.tail == 'true':
    x = sh.grep(sh.tail(args.stats, '-n', '1000'), tstr)
else:
    x = sh.grep(sh.grep("system.cpu.committedInsts::0 *200[0-9]\{6\}",
                        args.stats, '-m', "1", '-A', '600', '-B', '600'), tstr)

print tstr

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
    elif line.startswith('system.cpu.iew.exec_branches::0'):
        d['hptBranch'] = line.split()[1]
    elif line.startswith('system.cpu.iew.branchMispredicts::0'):
        d['hptMiss'] = line.split()[1]

base = float(d['system.cpu.fmt.numBaseSlots::0'])
wait = float(d['system.cpu.fmt.numWaitSlots::0'])
miss = float(d['system.cpu.fmt.numMissSlots::0'])
cycle = float(d['system.cpu.numCycles'])
ipc = float(d['system.cpu.ipc::0'])

print (base + wait + miss)/ (8*cycle)

qos = (base + miss) / (base + miss + wait)

print "Predicted QoS:", qos

print 'Predicted ST IPC:', ipc/qos

branch = float(d['hptBranch'])
branch_miss = float(d['hptMiss'])

print "Branch misprediction rate:", branch_miss / branch

