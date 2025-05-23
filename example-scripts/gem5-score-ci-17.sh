set -x

ulimit -n 4096

example_stats_dir=$1
json_path=$2

mkdir -p results

tag="gem5-score-example"
python3 batch.py -s $example_stats_dir -o results/$tag.csv

python3 simpoint_cpt/compute_weighted.py \
    -r results/$tag.csv \
    -j $json_path \
    --score results/$tag-score.csv \
    -v 17
