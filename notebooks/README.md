# Notebooks

这个目录是交互式实验入口。根目录的 CLI 和可复现实验说明见 [../README.md](../README.md)，高层计划见 [../experiment_pipeline_plan.md](../experiment_pipeline_plan.md) 和 [../synthetic_transformer_experiment_plan.md](../synthetic_transformer_experiment_plan.md)。

## 目录

```text
notebooks/
  Dyck_Syn_to_Rea/       从 synthetic Dyck counting 连接到 realistic NIAH 猜想
  Dyck_Synthetic/        Dyck / Shuffle Dyck / Dyck-k / CFG next-token notebooks
```

## Syn-to-Real

| Notebook | 状态 | 目的 | 主要产物 |
|---|---|---|---|
| `Dyck_Syn_to_Rea/Task_A_Length_Noise.ipynb` | 已实现 | 对应 synthetic plan 的 Task A：长度和噪声如何影响 Transformer counting feature 与行为表现 | `results/dyck_counter_task_a_summary.csv`, `figures/dyck_counter_task_a/`, `figures/dyck_counter_task_a_extra_probes/`, `figures/dyck_counter_task_a_ablation/`, `figures/dyck_counter_task_a_activation_patch/` |

相关脚本：

```bash
python scripts/summarize_task_a_results.py
python scripts/task_a_extra_probes.py
python scripts/task_a_height_ablation.py
python scripts/task_a_layerwise_activation_patch.py --from-existing
```

## Synthetic Dyck Notebooks

| Task | Notebook | 状态 | 说明 |
|---|---|---|---|
| Dyck | `Dyck_Synthetic/Dyck_4Models_Probe_No_Noise.ipynb` | 可运行 | 标准 Dyck no-noise 四模型 probe |
| Dyck | `Dyck_Synthetic/Dyck_4Models_Probe_50_Noise.ipynb` | 可运行 | 标准 Dyck 50% noise 四模型 probe |
| Dyck | `Dyck_Synthetic/Dyck_4Models_Probe_No_Noise_Long.ipynb` | 可运行 | 长 Dyck setting 和几何分析 |
| Shuffle Dyck | `Dyck_Synthetic/Shuffle_Dyck_4Models_Probe_No_Noise.ipynb` | 可运行 | 多 bracket stream 的 no-noise counter probe |
| Shuffle Dyck | `Dyck_Synthetic/Shuffle_Dyck_4Models_Probe_50_Noise.ipynb` | 可运行 | Shuffle Dyck 50% noise |
| Dyck-k | `Dyck_Synthetic/Dyck_k_4Models_Probe_No_Noise.ipynb` | 可运行 | 多 bracket type 的 stack-sensitive Dyck-k probe |
| Dyck-k | `Dyck_Synthetic/Dyck_k_4Models_Probe_50_Noise.ipynb` | 可运行 | Dyck-k 50% noise |
| Dyck-k | `Dyck_Synthetic/Dyck_k_4Models_Probe_No_Noise_Long.ipynb` | 可运行 | 长 Dyck-k setting |
| Dyck-(2,3) CFG | `Dyck_Synthetic/Dyck23_CFG_4Models_Probe_Next_Token.ipynb` | 可运行 | CFG-only valid strings，next-token prediction，length bins 0-40 / 42-80 / 82-120 |
| Dyck-(2,3) CFG | `Dyck_Synthetic/Dyck23_CFG_4Models_Probe_Next_Token_50_Noise.ipynb` | 可运行 | Dyck-(2,3) CFG next-token with noise setting |
| Dyck-(2,5) CFG | `Dyck_Synthetic/Dyck25_CFG_4Models_Probe_Next_Token.ipynb` | 可运行 | Dyck-(2,5) CFG next-token probe |
| Dyck-2 CFG | `Dyck_Synthetic/Dyck2_CFG_Transformer_Probe_Next_Token_Len_0_400.ipynb` | 可运行 | Transformer-only Dyck-2 CFG length 0-400 |
| Dyck-2 CFG | `Dyck_Synthetic/Dyck2_CFG_Transformer_Probe_Next_Token_Fixed_Len_400.ipynb` | 可运行 | Transformer-only Dyck-2 CFG fixed length 400 |
| Dyck-2 CFG | `Dyck_Synthetic/Dyck2_CFG_Transformer_Probe_Next_Token_Len_0_2000.ipynb` | 可运行 | Transformer-only Dyck-2 CFG length 0-2000 |
| Dyck-2 Markov | `Dyck_Synthetic/Dyck2_Markov_Transformer_Probe_Next_Token_Len_0_400.ipynb` | 可运行 | Markov-style Dyck-2 next-token comparison |

## Plan 中暂空的 notebooks

| Plan 项 | Notebook | 产物 |
|---|---|---|
| Task B: checkpoint emergence analysis |  |  |
| Task C: direct final count / JSON-like count answer |  |  |
| Task D: NeedleCount-synthetic bridge |  |  |
| Task E: CoT / reasoning-token counting |  |  |
| Markov / HMM / VoMC hidden-state task |  |  |
| Needle in a Haystack realistic task |  |  |

## 运行约定

- 推荐使用 `hse` kernel。
- notebook 中的长期产物应写到 `results/`、`figures/` 或 `paper_figs/`。
- 四模型 notebook 中，`rnn`、`lstm`、`transformer` 是稳定 baseline；`mamba` 依赖官方 `mamba-ssm`。
- `mamba_like` 只适合工程 smoke check，不作为正式 Mamba 结果。
- 如果 notebook 只是展示已有结果，优先用脚本更新 CSV/PNG，再在 notebook 中读取这些产物。
