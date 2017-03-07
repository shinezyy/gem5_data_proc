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
    #'committedInsts::1',
    'Slots::0',
    'ipc::',
    #'HPT',
    'numCycles',
    #'Overlapped',
    #'rectified',
    #'wait_to_miss::0',
    #'system.cpu.iew.exec_branches::0',
    #'system.cpu.iew.branchMispredicts::0',
    #'system.cpu.rename.*Cycles::0',
    #'system.cpu.itb.fetch_misses',
    #'system.cpu.itb.fetch_accesses',
    'system.cpu.*_Slots',
    'system.cpu.rename.*Full',
    'system.cpu.*_utilization',
    'overall_miss_rate',
]

def shorter(x):
    for line in str(x).split('\n'):
        first_half = line.split('#')[0].rstrip(' ')
        print first_half


tstr = 'Qos'
for s in targets:
    tstr += '\|'+s

if args.tail == 'true':
    x = sh.grep(sh.tail(args.stats, '-n', '2000'), tstr)
else:
    x = sh.grep(sh.grep("system.cpu.committedInsts::0 *2[0-9]\{8\}",
                        args.stats, '-m', "1", '-A', '1000', '-B', '600'), tstr)

print tstr

shorter(x)

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
    elif line.startswith('system.cpu.itb.fetch_accesses::0'):
        itb_all = float(line.split()[1])
    elif line.startswith('system.cpu.itb.fetch_misses::0'):
        itb_miss = float(line.split()[1])

base = float(d['system.cpu.fmt.numBaseSlots::0'])
wait = float(d['system.cpu.fmt.numWaitSlots::0'])
miss = float(d['system.cpu.fmt.numMissSlots::0'])
cycle = float(d['system.cpu.numCycles'])
ipc = float(d['system.cpu.ipc::0'])

print (base + wait + miss)/ (8*cycle)

qos = (base + miss) / (base + miss + wait)

print "Predicted QoS:", qos

print 'Predicted ST IPC:', ipc/qos

# print 'ITLB miss rate:', float(itb_miss) / itb_all

#  branch misPrediction Rate
'''
branch = float(d['hptBranch'])
branch_miss = float(d['hptMiss'])
print "Branch misprediction rate:", branch_miss / branch
'''

print ''

# rename cycles

'''
for line in str(x).split('\n'):
    if line.startswith('system.cpu.rename.'):
        first_half = line.split('#')[0].rstrip(' ')
        number = first_half.split(' ')[-1]
        print number
'''









