# Hidden State Evaluation

这个仓库用于研究序列模型 hidden state 是否编码任务所需的 sufficient statistics，以及模型是否在保留任务相关信息的同时压缩无关前缀信息。

当前主线有两层：

- 通用 pipeline：`task -> model -> train -> hidden-state extraction -> probes -> geometry/compression`，见 [experiment_pipeline_plan.md](experiment_pipeline_plan.md)。
- Transformer synthetic counting 实验：重点研究长上下文、噪声、counting feature emergence、hidden count 到输出的 gap，见 [synthetic_transformer_experiment_plan.md](synthetic_transformer_experiment_plan.md)。

## 当前状态

| 模块 | 状态 | 入口 | 主要产物 |
|---|---|---|---|
| Dyck baseline | 已实现 | `configs/dyck_no_noise.yaml`, `configs/dyck_noise.yaml` | `results/dyck_no_noise/`, `results/dyck_noise/` |
| Shuffle Dyck | 已实现 | `configs/shuffle_dyck.yaml`, `configs/shuffle_dyck_noise.yaml` | `results/notebooks/shuffle_dyck_reference_style/` |
| Dyck-k | 已实现 | `configs/dyck_k_no_noise.yaml`, `configs/dyck_k_50_noise.yaml`, `configs/dyck_k_long_no_noise.yaml` | `results/notebooks/dyck_k_reference_style/` |
| Dyck-(2,3) / Dyck-2 CFG next-token | 已实现 | `configs/dyck23_cfg_next_token.yaml`, `scripts/run_dyck23.py` | `results/dyck23_cfg_next_token/`, `results/dyck2_cfg_transformer_next_token_*` |
| Task A: length/noise Transformer sweep | 已实现 | `configs/generated_task_a/*.yaml`, `notebooks/Dyck_Syn_to_Rea/Task_A_Length_Noise.ipynb` | `results/dyck_counter_task_a_*`, `figures/dyck_counter_task_a*` |
| Task A extra probes / ablations / activation patching | 已实现 | `scripts/task_a_extra_probes.py`, `scripts/task_a_height_ablation.py`, `scripts/task_a_layerwise_activation_patch.py` | `results/dyck_counter_task_a_extra_probes/`, `results/dyck_counter_task_a_ablation/`, `results/dyck_counter_task_a_activation_patch/` |
| Task B: checkpoint emergence | 部分实现 | `configs/dyck_counter_emergence_transformer.yaml`, `scripts/run_pipeline.py` | `results/dyck_counter_emergence_transformer/` |
| Task C: direct final count / readout alignment | 未实现 |  |  |
| Task D: NeedleCount-synthetic bridge | 未实现 |  |  |
| Task E: CoT / reasoning-token counting | 未实现 |  |  |
| Markov / HMM / VoMC tasks | 未实现 |  |  |
| Needle in a Haystack task module | 未实现 |  |  |

未完成项先保留空入口和空产物列，避免把 plan 写成已经完成的结果。

## 安装

Conda 环境：

```bash
conda env update -f environment.yml
conda activate hse
python -m ipykernel install --user --name hse
```

或只安装 Python 包：

```bash
pip install -e .
```

核心依赖在 [pyproject.toml](pyproject.toml) 和 [environment.yml](environment.yml) 中声明。官方 Mamba 需要额外安装 `mamba-ssm`；没有官方包时，普通 pipeline 会跳过 `mamba`，但正式四模型对比不应把 `mamba_like` 当作替代结果。

## 常用命令

跑一个完整 train/extract/probe/geometry 闭环：

```bash
python scripts/run_pipeline.py --config configs/dyck_no_noise.yaml --model rnn --seed 0
```

只跑某个阶段：

```bash
python scripts/run_pipeline.py --config configs/dyck_no_noise.yaml --model rnn --seed 0 --stage train
python scripts/run_pipeline.py --config configs/dyck_no_noise.yaml --model rnn --seed 0 --stage extract
python scripts/run_pipeline.py --config configs/dyck_no_noise.yaml --model rnn --seed 0 --stage probe
python scripts/run_pipeline.py --config configs/dyck_no_noise.yaml --model rnn --seed 0 --stage geometry
```

跑 Task A 的一个 Transformer setting：

```bash
python scripts/run_pipeline.py --config configs/generated_task_a/dyck_counter_task_a_sparse_long.yaml --model transformer --seed 0
```

汇总 Task A notebook 和图：

```bash
python scripts/summarize_task_a_results.py
python scripts/task_a_extra_probes.py
python scripts/task_a_height_ablation.py
python scripts/task_a_layerwise_activation_patch.py --from-existing
```

跑 Dyck-(2,3) CFG next-token 实验：

```bash
python scripts/run_dyck23.py --models rnn lstm transformer --seed 0
```

## 目录结构

```text
configs/                         YAML 实验配置
configs/generated_task_a/         Task A length/noise sweep 的具体 setting
docs/                            工作流说明和论文/笔记材料
figures/                         脚本生成的分析图
notebooks/                       交互式实验 notebook，详见 notebooks/README.md
paper_figs/                      面向论文整理的图
results/                         训练、hidden state、probe、summary 和 HTML 结果
scripts/                         CLI 入口和 Task A 分析脚本
src/hse/                         Python package
tests/                           轻量 scaffold 测试
experiment_pipeline_plan.md       通用 sufficient-statistics/compression 计划
synthetic_transformer_experiment_plan.md  Transformer synthetic counting 计划
```

关键源码：

```text
src/hse/models/simple.py          RNN, LSTM, Transformer, official/fallback Mamba wrappers
src/hse/tasks/registry.py         任务注册表
src/hse/tasks/dyck/               Dyck sampler, labels, metrics
src/hse/tasks/shuffle_dyck/       Shuffle Dyck sampler and labels
src/hse/tasks/dyck_k/             Dyck-k stack task sampler and labels
src/hse/tasks/dyck23/             Dyck-(2,3) CFG sampler and labels
src/hse/tasks/markov/             占位
src/hse/tasks/needle/             占位
src/hse/analysis/probes/          Ridge/logistic probes and Dyck probe helpers
src/hse/analysis/compression/     Relevant retention / irrelevant forgetting probes
src/hse/analysis/geometry/        Probe-direction geometry checks
src/hse/experiments/              Notebook-friendly orchestration helpers
src/hse/utils/                    Config, IO, training, extraction utilities
```

## Pipeline 产物约定

一个标准 run 通常写到：

```text
results/<experiment_name>/<model>_seed<seed>/
  config.json
  metrics.json
  checkpoints/
  hidden_states/
  probes/
```

常见 probe 输出：

```text
probes/layerwise_probe.csv
probes/checkpoint_probe.csv
probes/summary.json
probes/compression_probe_rows.csv
probes/directions/
```

Task A 的汇总和额外分析会额外写到：

```text
results/dyck_counter_task_a_summary.csv
results/dyck_counter_task_a_extra_probes/
results/dyck_counter_task_a_ablation/
results/dyck_counter_task_a_activation_patch/
figures/dyck_counter_task_a*/
```

## Synthetic Transformer plan 对照

| Plan 项 | 当前落地 | 空缺 |
|---|---|---|
| A. 长度和噪声如何影响 counting | Task A configs、Transformer runs、summary notebook、extra probes、height ablation、activation patching 已有 | 多 seed 主实验仍需系统补齐 |
| B. counting feature 什么时候出现 | checkpoint 配置和 pipeline 支持已在 `dyck_counter_emergence_transformer.yaml` / `run_pipeline.py` 中准备 | `analyze_emergence.py`、`probe_emergence_step`、`behavior_emergence_step`、`verbalization_lag` 汇总还未写 |
| C. hidden count 到输出展示 | Task A extra probes 已有部分 output-head 使用分析 | direct final count task、NUM token answer、unembedding alignment/logit lens 主线为空 |
| D. Synthetic NIAH bridge |  |  |
| E. CoT / reasoning-token counting |  |  |

## Notebook 入口

主要 notebook 分两组：

- `notebooks/Dyck_Syn_to_Rea/Task_A_Length_Noise.ipynb`：Transformer Task A 长度/噪声分析。
- `notebooks/Dyck_Synthetic/`：Dyck、Shuffle Dyck、Dyck-k、Dyck CFG next-token 的 synthetic notebooks。

完整清单见 [notebooks/README.md](notebooks/README.md)。
