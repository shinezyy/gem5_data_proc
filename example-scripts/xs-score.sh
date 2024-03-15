set -x

ulimit -n 4096

export PYTHONPATH=`pwd`

# example_stats_dir=/nfs-nvme/home/share/wkf/SPEC06_EmuTasks_1215_allbump
# example_stats_dir=/nfs/home/share/wulingyun/fuck_perf/SPEC06_EmuTasks_new_spec_padding66_1.0_jemalloc_peak_bwaves_2
example_stats_dir=/nfs/home/share/liyanqin/xs-perf/2403-perf-pf/SPEC06_EmuTasks_0313_0139

mkdir -p results

tag="xs-fp"

python3 batch.py -s $example_stats_dir -o results/$tag.csv -X
python3 simpoint_cpt/compute_weighted.py --fp-only \
    -r results/$tag.csv \
    -j /nfs/home/share/jiaxiaoyu/simpoint_checkpoint_archive/spec06_rv64gcb_O3_20m_gcc12.2.0-intFpcOff-jeMalloc/checkpoint-0-0-0/cluster-0-0.json \
    --score results/$tag-score.csv
