template = '../run_gem5_alpha_spec06_benchmark.sh -b "{}\;{}"' + \
        ' -o {}/{} -s --smt -v fast -a ALPHA_{} -c {}'

confs = {
    'dyn' : 'dyn.py',
    'cc' : 'cc.py',
    'fc' : 'fc.py',
    'st' : 'sim_st.py',
}

conf = confs['dyn']

#odir = '~/hard'
#odir = '~/sim_st_64_lsq'
odir = '~/dyn_64_lsq'

inf = 'rand.txt'
#inf = 'sim_st.txt'
#inf = 'hard.txt'
#inf = 'qos.txt'

with open(inf) as f, open('./select.sh', 'w') as of:
    for line in f:
        t1, t2 = line.split()
        cmd = template.format(t1, t2, odir, t1 + '_' + t2,
                              'CC', confs['dyn'])
        of.write(cmd+'\n')
        print cmd


print 'generate run scripts from', inf, 'with conf', conf, 'to dir', odir
