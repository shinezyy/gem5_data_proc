import argparse
import sh

parser = argparse.ArgumentParser(description='check sum of slots')

parser.add_argument('stats', metavar='stats',
                    help='stats.txt to be converted')

args = parser.parse_args()

x = sh.tail(args.stats, '-n', "120")

table = []

for line in x:
    if 'fu_full' in line and float(line.split()[2].rstrip('%'))/100 > 0.01:
        table.append(line.split('#')[0])
    if 'fu_busy' in line:
        table.append(line.split('#')[0])

for line in table:
    print(line)
