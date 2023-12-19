# set -x

ulimit -n 4096

export PYTHONPATH=`pwd`


mkdir -p results

example_stats_dirs=(
/nfs-nvme/home/zhouyaoyang/projects/xs-gem5-eval/util/warmup_scripts/BOP-tag24
)
jsons=(
/nfs-nvme/home/share/checkpoints_profiles/spec06_rv64gcb_o3_20m_gcc12-fpcontr-off/json/o3_spec_fp_int-with-jemXalanc.json
)

len=${#jsons[@]}

for (( i=0; i<$len; i++ ))
do
    example_stats_dir=${example_stats_dirs[i]}
    tag=$(basename $example_stats_dir)
    python3 batch.py -s $example_stats_dir -o results/$tag.csv -F xalancbmk --eval-stat si_targets > /dev/null

    echo $tag
    python3 simpoint_cpt/compute_weighted.py \
        -r results/$tag.csv \
        -j ${jsons[i]} \
        -o results/$tag-weighted.csv
done
