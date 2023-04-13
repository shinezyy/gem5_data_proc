set -x

export PYTHONPATH=`pwd`

example_stats_dir=/nfs-nvme/home/share/zyy/gem5-results/example-outputs

mkdir -p results

python3 batch.py -s $example_stats_dir -t --topdown-raw -o results/example.csv

python3 simpoint_cpt/compute_weighted.py \
    -r results/example.csv \
    -j simpoint_cpt/resources/spec06_rv64gcb_o2_20m.json \
    -o results/example-weighted.csv

python3 simpoint_cpt/compute_weighted.py \
    -r results/example.csv \
    -j simpoint_cpt/resources/spec06_rv64gcb_o2_20m.json \
    --score results/example-score.csv
