template = '../run_gem5_alpha_spec06_benchmark.sh -b "{}\;{}"' + \
        ' -o {}/{} -s --smt -v fast -a ALPHA_DYN'

odir = '~/dyn_share_tlb2'


with open('rand.txt') as f:
    for line in f:
        t1, t2 = line.split()
        print template.format(t1, t2, odir, t1 + '_' + t2)

