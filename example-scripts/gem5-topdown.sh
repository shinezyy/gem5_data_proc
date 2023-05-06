set -x

ulimit -n 4096

export PYTHONPATH=`pwd`

# example_stats_dir=/nfs-nvme/home/share/tanghaojin/SPEC06_EmuTasks_topdown_0430_2023
example_stats_dir=/nfs-nvme/home/share/zyy/gem5-results/mutipref-replay-merge-tlb-pref

mkdir -p results

tag="gem5-topdown-example"
python3 batch.py -s $example_stats_dir -t --topdown-raw -o results/$tag.csv

python3 simpoint_cpt/compute_weighted.py \
    -r results/$tag.csv \
    -j simpoint_cpt/resources/spec06_rv64gcb_o2_20m.json \
    -o results/$tag-weighted.csv

python3 simpoint_cpt/compute_weighted.py \
    -r results/$tag.csv \
    -j simpoint_cpt/resources/spec06_rv64gcb_o2_20m.json \
    --score results/$tag-score.csv
