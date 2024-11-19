set -x

ulimit -n 4096

export PYTHONPATH=`pwd`

example_stats_dir=/nfs/home/jiaxiaoyu/emu_dual_core/xs-env/XiangShan/SPEC06_EmuTasks_1114_1453

mkdir -p results

tag="xs-dual-2024Nov"

for core in 0 1; do
    export XS_CORE_ID=$core
    python3 batch.py -s $example_stats_dir -o results/$tag-core$core.csv -X
    python3 simpoint_cpt/compute_weighted.py \
        -r results/$tag-core$core.csv \
        --score results/$tag-score-core$core.csv \
        -j /nfs/home/jiaxiaoyu/emu_dual_core/disable_timer/checkpoint-0-0-0/spec06_gcc_disable_dualcore_base_0.5coverage.json
done

