# 按需性能分析入口

`run.py` 继续用于一条命令生成 CSV、加权结果和分数。需要原始计数器搜索、
逐切片查询或 Dashboard 接入时，使用 `analysis_pipeline.py`；它复用 `batch.py`、
`utils.targets_yaml` 和 `simpoint_cpt.compute_weighted`，不引入新的算分算法或依赖。

职责边界：本仓库负责日志解析、profile 切片选择、原始计数器提取、加权、算分；
调用方负责来源定位、任务队列/取消、缓存身份与生命周期、查询筛选及展示。
入口没有 GitHub、HTTP 或 Dashboard 依赖。

## 准备快照

将请求保存为 `prepare.json`：

```json
{
  "source": "/path/to/ci-archive",
  "backend": "gem5",
  "profile_path": "/path/to/ci-archive/cluster.json",
  "output": "/tmp/analysis/snapshot-a",
  "spec": "06",
  "extract_jobs": 4
}
```

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 analysis_pipeline.py prepare prepare.json
```

- `source`：GEM5 为包含 `spec_all/` 的解包归档；RTL 为报告目录。
- `backend`：`gem5`（默认）或 `xs`。`profile_path` 默认为 `<source>/cluster.json`。
- `spec`：SPEC 版本 `06`、`17` 或 `26`，决定参考时间；当前真实数据验证覆盖 SPEC06 单线程。
- `extract_jobs`：1–16，默认 4。原始提取线程与 batch 进程分阶段执行。
- `manifest`：可选的来源信息对象，原样保留；完成计数、分数等结果字段由处理器生成。
- `stage_dir`：可选的临时目录。调用方传入时负责清理，适用于终止整个子进程组的场景；
  省略时由入口创建并在正常退出时清理。请求中的路径相对于调用命令时的工作目录解析。

RTL 例子只需改为：

```json
{
  "source": "/nfs/home/cirunner/perf-report/cr261105-a253c1163-DefaultConfig",
  "backend": "xs",
  "profile_path": "/nfs/home/share/checkpoints_profiles/spec06_gcc16_rva23_novec_260820/json/checkpoints_cov0.3.json",
  "output": "/tmp/analysis/rtl",
  "spec": "06"
}
```

GEM5 仅接纳有 `completed`、无 `abort`、存在 warmup 与完整最终 stats dump 的切片。
RTL 按 profile 的 workload、point、weight 选择目录，接受精确权重、六位小数权重或无权重目录；
同一个切片有多个匹配目录时拒绝自动选择。目录选择与 `batch.py --json-filter` 共用规则。
RTL 还要求 `simulator_out.txt` 有成功结束标记，无 BAD TRAP / Assertion / ABORT，
且存在两个不同时间的单核 commitInstr dump。尚不支持多核和 SMT 成对 workload。
两种来源均要求 cycles、committedInsts、CPI 为正的有限值。

原始缓存使用最终测量窗口；RTL 原始键为 `<完整层次>:<counter>`，重复同名键记为 null。
YAML 提取读取同一份日志的本地副本：GEM5 只保留最终窗口，RTL 保留完整原文，
以维持现有解析器的多次匹配累加规则。这里没有更改现有 YAML 指标口径。

输出目录应由调用方专用，包含：

| 文件 | 内容 |
| --- | --- |
| `points.csv` | 有效切片的 YAML 指标 |
| `weighted.csv` | 现有加权器的 benchmark 结果 |
| `score.csv` | 现有 SPEC 算分结果 |
| `cluster.json` | 本次使用的 profile 副本 |
| `raw/<workload>_<point>.json.gz` | 最终窗口全部原始数值计数器 |
| `catalog.json` | YAML groups、别名及 raw 名称 |
| `manifest.json` | 来源、有效切片、排除原因、完成/期望数量、分数和 schema version |
| `progress.json` | 原子替换的阶段及切片进度 |

`manifest.json` 最后生成。非零退出表示失败，调用方应丢弃本次未完成输出，
并在处理前后检查输入与处理器身份，避免缓存正在被覆盖的归档。
入口不管理取消、缓存目录或并发写入；不要让多个任务共用输出目录。

## 查询快照

```json
{
  "snapshots": ["/tmp/analysis/snapshot-a", "/tmp/analysis/rtl"],
  "metrics": ["cpi", "ipc", "raw:system.cpu.numCycles"],
  "output": "/tmp/analysis/query"
}
```

```bash
python3 analysis_pipeline.py compare compare.json
```

输出 `values.json`，按 snapshots 顺序包含 `benchmark` 和 `simpoint` 行，行标识为 `name`。
YAML 指标使用别名，原始计数器使用 `raw:` 前缀。任意 raw 项仍交给现有加权器处理。
加权 IPC 为 `1 / weighted_CPI`；缺失和非有限值输出 JSON null，不补零。
跨 RTL/GEM5 的同名 YAML 指标需要调用方提示口径差异；raw 名称不自动映射。
该入口不按查询筛选重算套件分数，也不做 A/B 百分比、排序或分页。

## 验证

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

`test_analysis_pipeline.py` 覆盖混合 profile 目录、权重格式与歧义、最终 epoch raw、
多核/未完成拒绝、原版 RTL 累加行为，以及两种来源的完整 CLI 提取与缺失 raw 加权。
