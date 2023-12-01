set -x

ulimit -n 4096

export PYTHONPATH=`pwd`

example_stats_dir=/nfs-nvme/home/zhouyaoyang/projects/xs-gem5-eval/util/warmup_scripts/BOP-tag12

mkdir -p results

tag="gem5-traffic-example"
python3 batch.py -s $example_stats_dir --cache --eval-stat mem_targets -o results/$tag.csv

python3 simpoint_cpt/compute_weighted.py \
    -r results/$tag.csv \
    -j /nfs-nvme/home/share/checkpoints_profiles/spec06_rv64gcb_o3_20m_gcc12-fpcontr-off/json/o3_spec_fp_int-with-jemXalanc.json \
    -o results/$tag-weighted.csv
