# template = '../run_gem5_alpha_spec06_benchmark.sh -b "{}\;{}"' + \
#        ' -o {}/{} -s --smt -v fast -a ALPHA_{}'# -c {}'

template = '$gem5_root/create_simpoint.sh -b "{}"' + \
        ' -o $gem5_root/simpoint/{} -a ALPHA_CC -c simpoint.py'

confs = {
    'dyn' : 'dyn.py',
    'cc' : 'cc.py',
    'fc' : 'fc.py',
    'st' : 'sim_st.py',
    'sp' : 'simpoint.py',
}

#conf = confs['dyn']

#odir = '~/hard'
#odir = '~/sim_st_64_lsq'
#odir = '~/dyn_64_lsq_2'
#odir = '~/hard_dyn'
#odir = '~/hard_cc_70'
#odir = '~/old_cc_80'

#inf = 'rand.txt'
#inf = 'sim_st.txt'
#inf = 'hard.txt'
#inf = 'qos.txt'
inf = './all_spec.txt'

with open(inf) as f, open('./all_simpoint.sh', 'w') as of:
    for line in f:
        line = line.strip()
        #t1, t2 = line.split()
        #cmd = template.format(t1, t2, odir, t1 + '_' + t2,
        #                      'CC', conf)
        cmd = template.format(line, line)
        of.write(cmd+'\n')
        print(cmd)


print('generate run scripts from', inf) # , 'with conf', conf, 'to dir', odir
