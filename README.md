This is a tool to extract GEM5 & XS performance counter from the output of GEM5 & XS simulation.
# Examples to extract GEM5 & XS performance counter

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

# One-shot runner (recommended)

Use `run.py` to extract CSV and compute weighted + score in one command.
It auto-detects XS format by searching `simulator_err.txt` in the directory.
By default it enables all YAML groups (equivalent to `batch.py --groups all`).
For CI result directories, it reads `benchmark_type` from `metadata.txt` and
selects the corresponding built-in SimPoint profile.

``` shell
python3 run.py /path/to/results/tag --out-dir results
python3 run.py /path/to/results/tag --out-dir results --benchmark-type gcc12  # gcc12/SMT
python3 run.py /path/to/results/tag --out-dir results --benchmark-type gcc16
python3 run.py /path/to/results/tag --out-dir results -j /path/to/cluster.json
python3 run.py /path/to/results/tag --out-dir results -g basic,branch,tage  # override groups
python3 run.py /path/to/results/tag --out-dir results --jobs 32  # extraction workers, default auto
```

Notes:
- A `benchmark_type` containing `gcc12`, `gcc15`, `gcc16`, or `xscc` maps to
  the corresponding profile, regardless of other prefixes or suffixes.
- GCC15 and GCC16 presets use their full 1.0c JSON; a 0.3c result is weighted
  using the matching points present in that result.
- `--benchmark-type {gcc12,gcc15,gcc16,xscc}` overrides metadata detection.
  The old `--slice` spelling remains available as an alias.
- Without metadata or an explicit selection, the default profile is GCC16.
- `-j /path/to/cluster.json` overrides `--benchmark-type`.
- For XS / RTL directories, `run.py` automatically reuses the selected json as
  `batch.py --json-filter`, so mixed-profile directories are filtered to the
  intended SimPoint slice before weighting and scoring. Flattened directories
  with a weight suffix must match the full `workload_point_weight` entry.

RTL extraction requires positive `committedInsts` and `cycles`, plus warmup and
measurement `commitInstr` dumps. Missing analysis counters are kept as `NaN`
(including dependent metrics), with one warning summary per missing-counter set
and one example path. Empty extraction stops before weighting. Use `-g basic`
when only IPC and score are needed.

RTL Intel Topdown uses ROB `total_flush` for total redirects and measured
`committedInsts` for retiring (DefaultConfig width=8). Recent RTL does not export
the legacy `inst_spec` / `recovery_bubble` counters, so bad-speculation metrics
and `backendBound` remain `NaN`. Rename `recovery_stall` only counts RAB walks
and is not a substitute; full Topdown accounting needs consistent RTL slot units.

The legacy wrapper is still available (it calls `run.py` internally):
``` shell
bash example-scripts/gem5-topdown-tag.sh /path/to/results/tag
bash example-scripts/gem5-topdown-tag.sh /path/to/results/tag -g basic,branch,tage,mbtb
```

# Reusable analysis snapshots

For raw-counter search and per-SimPoint queries, use `analysis_pipeline.py prepare|compare request.json`.
It reuses `batch.py` and the existing weighting/scoring implementation; callers own source resolution,
queue/cancellation, and cache lifecycle. See [the request/output contract](docs/analysis_pipeline.md).
The existing `run.py` workflow remains available.

# YAML targets (recommended for adding new counters)

Targets can be defined in `targets/*.yaml` and enabled by group name.
This is the preferred way to add new counters without editing `utils/target_stats.py`.

List groups:
``` shell
python3 batch.py --list-groups
```

Enable YAML groups (missing counters are allowed and kept as NaN):
``` shell
python3 batch.py -s /path/to/results --groups basic,branch,tage -o results/run.csv
python3 batch.py -s /path/to/results --groups all -o results/run.csv
python3 run.py /path/to/results --out-dir results -g basic,branch,tage
```

Common extra groups:
``` shell
python3 run.py /path/to/results --out-dir results -g basic,branch,fetch
python3 run.py /path/to/results --out-dir results -g basic,intel_topdown
python3 run.py /path/to/results --out-dir results -g basic,l1_demand_read,l1_prefetch,l2_demand,l2_prefetch
python3 run.py /path/to/results --out-dir results -g basic,mem_lsq,mem_violation,mem_lsq_pipe,mem_sbuffer,mem_tlb,mem_dcache
```

Prefetch notes:
- `l1_demand_read` normalizes GEM5 L1 demand-read miss/access counters against
  XS `LoadUnit_0/1/2` sums.
- `l1_prefetch` groups both L1 prefetcher issued/useful counters and the
  `l2_l1pf_*` counters, because those L2 counters describe requests originating
  from L1 prefetchers.
- `l2_demand` groups L2 load/store demand counters.
- `l2_prefetch` normalizes GEM5 BOP counters against XS `BOP + PBOP`.
- The normalization happens automatically inside `batch.py` after YAML
  extraction, so the output CSV includes both raw helper columns and comparable
  semantic columns such as `l1_read_*` and `l2_bop_*`.

Local (not committed) extensions can be put under `targets/local/*.yaml` (gitignored).

Use `gem5_regex` when one output column must match an explicit regular
expression instead of the usual stat-name/glob conversion. For example, the
following extracts the scalar counter from a single-thread run and only the
aggregate `::total` counter from an SMT run:

``` yaml
groups:
  basic:
    gem5_regex:
      committedInsts: 'system\\.cpu\\.committedInsts(?:::total)?'
```

The regular expression must match the complete stat name. The parser appends
the whitespace/value capture used to read `stats.txt`.

Use `derived_gem5` or `derived_xs` when the two simulators need different raw
counter combinations before a common derived metric can be evaluated. The
backend-specific expressions run before `derived`:

``` yaml
groups:
  branch_source:
    gem5:
      s1WrongAbtb: s1PredWrongAbtb
    xs:
      s1WrongAbtbRaw: commit_branch_mispredicts_s1_source_Abtb
      s1WrongAbtbUtage: commit_branch_mispredicts_s1_source_AbtbUtage
    derived_xs:
      s1WrongAbtb: s1WrongAbtbRaw + s1WrongAbtbUtage
    derived:
      s1WrongAbtbMPKI: s1WrongAbtb * 1000 / committedInsts
```

Duplicate YAML keys are rejected instead of being silently overwritten.

# Compare weighted CSV (web UI)

Compare two weighted CSVs in a local web UI:
``` shell
python3 compare_weighted.py results/a-weighted.csv results/b-weighted.csv
```

Features:
- Group filter (multi-select, union): show one or more YAML groups at a time
- Click `x` in a column header to hide a column; `reset hidden` to restore
- `only changed` to show columns with any diff
- `export csv` to export current view as diff strings: `+12.34% (1.23 -> 1.38)`
# Example for eval targets

The `eval targets` trick makes use of Python's `eval` to avoid creating new options for every new stat group

Use eval target to extract GEM5's memory bandwidth:
``` shell
batch.py -s /path/to/results/top/directory --eval-stat mem_targets
```

Using eval target to extract GEM5's memory bandwidth and memory dependency counters:

``` shell
batch.py -s /path/to/results/top/directory --eval-stat mem_targets#mem_dep_targets
```

Using eval target to extract XS's memory bandwidth and memory dependency counters:

``` shell
batch.py -s /path/to/results/top/directory -X --eval-stat mem_targets#mem_dep_targets
```

# Compute weighted performance

**Unified weighted metric computation with batch.py**

Now we use `batch.py` to compute the performance for each checkpoint.
Then we use simpoint_cpt/compute_weighted.py to compute **weighted metrics** and **scores**
Example usage here:
``` shell
export PYTHONPATH=`pwd`

example_stats_dir=/nfs-nvme/home/share/zyy/gem5-results/example-outputs

mkdir -p results

python3 batch.py -s $example_stats_dir -t --topdown-raw -o results/example.csv  # The topdown results for each checkpoint

python3 simpoint_cpt/compute_weighted.py \
    -r results/example.csv \
    -j simpoint_cpt/resources/spec06_rv64gcb_o2_20m.json \
    -o results/example-weighted.csv  # The weighted topdown counters for each benchmark

python3 simpoint_cpt/compute_weighted.py \
    -r results/example.csv \
    -j simpoint_cpt/resources/spec06_rv64gcb_o2_20m.json \
    --score results/example-score.csv  # The SPEC score for each benchmark and overll score

```

## Analysis topdown performance

First, we need to get the topdown outputs for one tests
```
bash example-scripts/gem5-topdown-tag.sh spec_ideal_numBr6
```

Then, we can use `topdown/draw_new.py` to analyze the topdown performance, and draw pictures, save to `figure/`
```
python3 topdown/draw_new.py -f=1 -p  -t1=spec_ideal_numBr4 -t2=spec_ideal_numBr6
# -f=1 means the highest level of detail
# -p means print the level 1 percentage and diff two tags outputs
# -t1=spec_ideal_numBr4 means the first tag
# -t2=spec_ideal_numBr6 means the second tag
```

```
python3 topdown/draw_new.py -f=3 -c=Frontend -t1=spec_ideal_numBr4 -t2=spec_ideal_numBr6
# -f=3 means the most detailed level
# -c=Frontend means the category, choises: Frontend, Backend, BadSpec
```

## Dual-core performance
stats parser will obtain XS_CORE_ID from environment variables to choose which core to compute score:

```
export XS_CORE_ID
python3 batch.py -s $example_stats_dir -o results/$tag-core$core.csv -X
```

Full scripts to obtain dual-core performance is in `example-scripts/xs-dual-core.sh`

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


## RTL / GEM5 dispatch accounting

Use the measured instruction window and distinguish accepted dispatch uops from
stall labels and issue attempts:

```bash
python3 run.py /path/to/results --out-dir results \
  -g basic,dispatch_accounting,dispatch_histogram,rtl_dispatch,rename_resources,intel_topdown,old_topdown
```

- `dispatch_uops`: GEM5 `iew.dispatchedInsts`; RTL Dispatch `in_fire_count`.
  Both count acceptance at the rename/dispatch boundary, including speculative
  work and eliminated instructions; they are not execution/replay counts.
- `retired_uops`: GEM5 `commit.opsCommitted`; RTL ROB `commitUop`.
  `committedInsts` remains the architectural instruction denominator. Instruction
  fusion and micro-op expansion can make the two retirement counts different.
- `dispatch_retire_delta_per_inst` is a finite-window proxy for non-retired work,
  not an exact wrong-path count: pipeline occupancy at the window boundaries and
  differences in uop modeling must still be considered.
- `dispatch_nostall_minus_fire` / `dispatch_nostall_fire_ratio` expose differences
  between NoStall labels and real dispatch. Do not substitute NoStall for fire.
- `dispatch_histogram_minus_fire` independently compares the histogram's weighted
  bin sum with acceptance counts. GEM5 sums bins 0..32; current RTL DefaultConfig
  sums 0..8. `dispatch_histogram_minus_cycles` checks sampled-cycle coverage.
  In the astar CI audit, GEM5 count residuals were exactly zero; RTL histogram
  counters differed by at most one width-8 cycle due to their sampling boundary.
  Missing bins remain NaN; do not interpret an unavailable check as passing.
- `rtl_dispatch` preserves the current RTL category names with a `rtl_dispatch_`
  prefix, including IntFlStall, RobStall, load cancellation and unclassified slots.
  `rtl_dispatch_partition_residual` checks the width-8 category sum against cycles;
  this formula is specific to DefaultConfig, not arbitrary widths or multi-core
  combined logs. BackendOtherCoreStall is unclassified, not a confirmed FU stall.
- `rename_resources` keeps backend-specific counters separate. In particular,
  zero GEM5 `fullRegistersEvents` does not prove absence of register starvation:
  the model can attribute it to a ROB/LSQ-head reason first. RTL rename stall
  cycles overlap dispatch labels and must not be added to them.

Intel Topdown now uses GEM5 **Commit** `branchMispredicts` with Commit
`totalSquash`; the old execution-stage count is retained as `iew_br_mis_pred`.
This avoids negative machineClears caused by mixing event stages. Both formats
use measured `committedInsts` instead of a fixed 20M instruction denominator.
The legacy GEM5 dispatch export also includes ControlRecovery, MemVioRecovery,
VPRecovery and TrapRecovery, which must not be silently omitted from its total.

These changes do not align every Intel Topdown event: GEM5 and RTL still use
few-ops thresholds 8 and 4, respectively, and different store-stall conditions.
RTL unavailable speculative/recovery counters stay NaN; `dispatch_uops` is not
silently substituted into legacy `inst_spec`. Raw core/memory fractions in this
processor are also distinct from gem5's scaled native backend fractions.

For benchmark comparisons, keep raw counters and CPI. A weighted average of
per-point percentages is not the same as a ratio of weighted counts. Rank
alignment slices by `profile_instructions * simpoint_weight * (RTL_CPI - GEM5_CPI)`
as well as runtime share; raw weights from different benchmark inputs are not
directly comparable.

Use `dispatch_histogram` together with `dispatch_accounting` for acceptance and
cycle checks; missing dependencies intentionally produce NaN. Intel Topdown
validation here is for single-thread width-8 runs. Accepting a scalar or
`::total` branch counter does not make the whole Intel group SMT-compatible:
per-thread-only totalSquash and other slot counters still need explicit
aggregation before drawing SMT conclusions.
