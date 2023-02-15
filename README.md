This is a tool to extract GEM5 & XS performance counter from the output of GEM5 & XS simulation.
# Extract GEM5 & XS performance counter

We use `batch.py` to extract the performance counter for each checkpoint.

Get full option list of `batch.py` with
``` shell
batch.py -h
```

To use `batch.py` anywhere, you can add `gem5_data_proc` to you PATH:
``` shell
export PATH='/path/to/gem5_data_proc':$PATH
```

Use `batch.py` to extract GEM5's cache performance counters:
``` shell
batch.py -s /path/to/results/top/directory  --cache -f stats.txt
```

Include only a specific benchmark like gromacs:
``` shell
batch.py -s /path/to/results/top/directory  --cache -f stats.txt -F gromacs
```

Use `batch.py` to extract XS's cache & branch performance counters:
``` shell
batch.py -s /path/to/results/top/directory --cache --branch --xiangshan -f simulator_err.txt
```

# Compute weighted performance

When the simulation results are gathered on SimPoint simulation points,
the weighted performance can be computed with
`weighted_extraction/weighted_ipc.py`.
Before running the script, please follow these steps:
First, set PYTHONPATH to the root of this project: ``export PYTHONPATH=`pwd` ``
Then modify the example function `xiangshan_spec2006` in `weighted_extraction/weighted_ipc.py`.

Confs contains configurations of a GEM5 or XS batch task and the path:
``` Python
confs = {
    "The-configuration-description": "/The/path/to/the/top/directory/of/the/batch/task",
}
```

The parameters of `compute_weighted_cpi` should be set as follows:
``` Python
    compute_weighted_cpi(
            ver=ver,  # version of SPEC CPU , '06' or '17'
            confs=confs,  # confs above
            base='XiangShan-Nanhu',  # sometime, we want to compute the relative performance, and we can pass a configuration as the base

            simpoints=f'/nfs-nvme/home/zhouyaoyang/projects/BatchTaskTemplate/resources/simpoint_cpt_desc/spec{ver}_rv64gcb_o2_20m.json',
            # simpoints contains the description of the SimPoint simulation points, including start instruction and weight in a json file

            prefix = 'xs_',
            # if extracting XS performance, the prefix should be 'xs_'; if extracting GEM5 performance, the prefix should be ''
            # `prefix` will be used in eval(), such as `eval(f"c.{prefix}add_branch_mispred(d)")`

            stat_file='simulator_err.txt',  # the file name of the output of GEM5 or XS; Usually, simulator_err.txt or err.txt for XS and stats.txt for GEM5

            insts_file_fmt =
            '/nfs-nvme/home/share/checkpoints_profiles/spec{}_rv64gcb_o2_20m/logs/profiling/{}.log',
            # The format of the path of the instruction count file of each benchmark, generated from NEMU when taking checkpoints.


            clock_rate = 2 * 10**9,  # simulation clock rate
            min_coverage = 0.98,  # the minimum coverage of the SimPoint simulation points, 
            merge_benckmark=True,  # merge multiple inputs for the same benchmark, like gcc_200 and gcc_166
            output_csv='nanhu-11-12.csv',  # output file name
            )
```

Then execute the script from the root of this project:
``` shell
python3 weighted_extraction/weighted_ipc.py
```

# How to add more interested stats

## Simple stats target group
See `cache_targets` defined in utils/target_stats.py and its usage in batch.py.

Simple stats target group contains **a list of targets**.
Each entry of the list is a **regex**.
`batch.py` will ``search'' for the pattern in given stats file,
and name it with the first match group in parentheses.
For example
``` regex
(l3\.demandAcc)esses::total'
        ^
        The first match group, used as name
```

## Complex stats target

Complex stats target group is a dictionary.
The key of an entry is the name of the target.
The value of an entry has two possible types: `list` or `str`.

If `str`, it is the regex to search.
(`xs_cache_targets_nanhu` in utils/target_stats.py is an example.)

If `list`, like `xs_cache_targets_22_04_nanhu`,
value[0] is the regex to search, while values[1] is how many times such pattern repeats.
This is to handle the case that one pattern repeats multiple times in specific version of XS.
The occurs because the performance counter of different banks of L2/L3 caches are named the same.
Because this is to handle the buggy behavior in RTL, this type of stats group is rarely used
and is out of maintained.


# Assumed directory structure 
A typical directory structure of GEM5 results looks like:

``` shell
.
|-- bwaves_1299
|   |-- completed
|   |-- dcache_miss.db
|   |-- dramsim3.json
|   |-- dramsim3.txt
|   |-- dramsim3epoch.json
|   |-- log.txt
|   `-- m5out
|       |-- TableHitCnt.txt
|       |-- altuseCnt.txt
|       |-- config.ini
|       |-- config.json
|       |-- misPredIndirect.txt
|       |-- misPredIndirectStream.txt
|       |-- missHistMap.txt
|       |-- stats.txt
|       |-- topMisPredictHist.txt
|       `-- topMisPredicts.txt
`-- gcc_2000
    |-- completed
    |-- dcache_miss.db
    |-- dramsim3.json
    |-- dramsim3.txt
    |-- dramsim3epoch.json
...
```

A typical directory structure of XS looks like:
```
.
|-- GemsFDTD_1041040000000_0.022405
|   |-- simulator_err.txt
|   `-- simulator_out.txt
|-- GemsFDTD_1121140000000_0.004928
|   |-- simulator_err.txt
|   `-- simulator_out.txt
|-- GemsFDTD_1175660000000_0.022268
|   |-- simulator_err.txt
|   `-- simulator_out.txt
...
```

