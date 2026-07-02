from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_EXPERIMENT_ORDER = [
    "dyck_counter_task_b_clean_short_smoke",
    "dyck_counter_task_b_noisy_short_smoke",
]
EXT_DIR = ROOT / "results" / "dyck_counter_task_b_extensions"
EXT_FIG_DIR = ROOT / "figures" / "dyck_counter_task_b_extensions"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", default=str(ROOT / "results" / "dyck_counter_task_b_training_dynamics"))
    parser.add_argument("--figure-dir", default=str(ROOT / "figures" / "dyck_counter_task_b_training_dynamics"))
    parser.add_argument("--output", default=str(ROOT / "notebooks" / "Dyck_Syn_to_Rea" / "Task_B_Training_Dynamics.ipynb"))
    args = parser.parse_args()

    summary_dir = Path(args.summary_dir)
    output = Path(args.output)
    summary = pd.read_csv(summary_dir / "emergence_summary.csv")
    best = pd.read_csv(summary_dir / "checkpoint_best_probe.csv")
    behavior = pd.read_csv(summary_dir / "behavior_log.csv")
    figures = pd.read_csv(summary_dir / "figures.csv")
    summary = order_summary(summary)
    summary = attach_behavior_max(summary, behavior)
    summary = attach_oracle_ceiling(summary)
    final_forced_free = compute_final_forced_free_behavior(summary)
    if not final_forced_free.empty:
        final_forced_free.to_csv(summary_dir / "final_forced_free_behavior.csv", index=False)

    cells = [
        md(
            "# Task B: Training Dynamics of the Dyck Counting Feature\n\n"
            "这个 notebook 对应核心实验 B：在训练过程中，Transformer 什么时候形成可线性读出的 counting feature，"
            "以及这个 feature 什么时候真正转化为 Dyck next-token 行为。\n\n"
            "当前版本先跑 smoke setting：clean-short 和 noisy-short。后续 long/sparse setting 可以用同一套脚本继续追加。"
        ),
        md(read_this_first(summary)),
        code(
            "from pathlib import Path\n"
            "import pandas as pd\n\n"
            "ROOT = Path.cwd()\n"
            "while ROOT != ROOT.parent and not (ROOT / 'results').exists():\n"
            "    ROOT = ROOT.parent\n"
            "SUMMARY_DIR = ROOT / 'results' / 'dyck_counter_task_b_training_dynamics'\n"
            "summary = pd.read_csv(SUMMARY_DIR / 'emergence_summary.csv')\n"
            "best = pd.read_csv(SUMMARY_DIR / 'checkpoint_best_probe.csv')\n"
            "behavior = pd.read_csv(SUMMARY_DIR / 'behavior_log.csv')\n"
            "summary = summary.merge(\n"
            "    behavior.groupby('run_key')['eval_dyck_acc'].max().rename('max_eval_dyck_acc').reset_index(),\n"
            "    on='run_key',\n"
            "    how='left',\n"
            ")\n"
            "summary"
        ),
        md(metric_definitions()),
        md("## Setting Comparison\n\n" + markdown_table(comparison_table(summary), max_rows=20)),
        md(final_forced_free_section(final_forced_free)),
    ]

    for section_index, (_, summary_row) in enumerate(summary.iterrows(), start=1):
        run_key = str(summary_row["run_key"])
        best_run = best.loc[best["run_key"] == run_key].copy()
        behavior_run = behavior.loc[behavior["run_key"] == run_key].copy()
        figure_rel = figures.loc[figures["run_key"] == run_key, "figure"].iloc[0]
        figure_notebook_rel = rel_from_notebook(output, ROOT / figure_rel)
        config = load_run_config(summary_row)
        title = setting_title(config, summary_row)

        best_table = best_run[
            [
                "checkpoint",
                "step",
                "checkpoint_order",
                "best_layer",
                "height_r2",
                "height_class_accuracy",
                "legal_next_class_accuracy",
                "eval_dyck_acc",
            ]
        ].sort_values(["checkpoint_order", "checkpoint"])
        best_table = best_table.drop(columns=["checkpoint_order"])
        behavior_tail = behavior_run[["step", "train_loss", "eval_loss", "eval_acc", "eval_dyck_acc"]].tail(12)

        cells.extend(
            [
                md(f"## {section_index}. {title}"),
                md(f"### {section_index}.1 实验设置\n\n" + setting_description(config, summary_row)),
                md(f"### {section_index}.2 Emergence summary\n\n" + markdown_table(pd.DataFrame([summary_row]), max_rows=1)),
                md(f"### {section_index}.3 Checkpoint-level best layer\n\n" + markdown_table(best_table, max_rows=30)),
                md(f"### {section_index}.4 Recent behavior log\n\n" + markdown_table(behavior_tail, max_rows=20)),
                md(
                    f"### {section_index}.5 Training dynamics figure\n\n"
                    f"![Task B training dynamics]({figure_notebook_rel})"
                ),
                md(f"### {section_index}.6 当前结果解读\n\n" + interpretation(summary_row, best_run, config)),
                md(f"### {section_index}.7 下一步检查\n\n" + followup_notes(summary_row, config)),
            ]
        )

    cells.extend(extension_cells(output))
    cells.append(md(overall_interpretation()))

    cells.append(
        code(
            "# 可选：重新汇总并生成 notebook\n"
            "# !python scripts/task_b_training_dynamics.py "
            "--experiments dyck_counter_task_b_clean_short_smoke dyck_counter_task_b_noisy_short_smoke\n"
            "# !python scripts/task_b_training_dynamics_extensions.py --device cuda --eval-examples 256 --eval-batch-size 32 --max-rows 20000\n"
            "# !python scripts/write_task_b_training_dynamics_notebook.py"
        )
    )

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {output}")


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def attach_behavior_max(summary: pd.DataFrame, behavior: pd.DataFrame) -> pd.DataFrame:
    if behavior.empty or "run_key" not in behavior or "eval_dyck_acc" not in behavior:
        return summary
    maxima = (
        behavior.groupby("run_key", as_index=False)["eval_dyck_acc"]
        .max()
        .rename(columns={"eval_dyck_acc": "max_eval_dyck_acc"})
    )
    return summary.merge(maxima, on="run_key", how="left")


def attach_oracle_ceiling(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        stats = task_oracle_ceiling(ROOT / str(row["run_dir"]))
        rows.append(stats)
    if not rows:
        return summary
    return pd.concat([summary.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def task_oracle_ceiling(run_dir: Path) -> dict[str, float]:
    labels_path = run_dir / "hidden_states" / "final" / "labels.parquet"
    config_path = run_dir / "config.json"
    if not labels_path.exists() or not config_path.exists():
        return {
            "oracle_eval_dyck_acc_ceiling": np.nan,
            "forced_target_fraction": np.nan,
            "free_target_fraction": np.nan,
        }
    config = json.loads(config_path.read_text(encoding="utf-8"))
    task = config.get("task", {})
    labels = pd.read_parquet(labels_path).sort_values(["example_id", "position"]).reset_index(drop=True)
    grouped = labels.groupby("example_id", sort=False)
    next_position = grouped["position"].shift(-1)
    has_target_label = next_position.eq(labels["position"] + 1)
    target_is_dyck = grouped["is_dyck_position"].shift(-1).fillna(False).astype(bool)
    target_mask = has_target_label.to_numpy(dtype=bool) & target_is_dyck.to_numpy(dtype=bool)
    if not target_mask.any():
        return {
            "oracle_eval_dyck_acc_ceiling": np.nan,
            "forced_target_fraction": np.nan,
            "free_target_fraction": np.nan,
        }

    total_length = int(task.get("total_length", labels["dyck_seen"].max()))
    max_opens = total_length // 2
    remaining_dyck = total_length - labels["dyck_seen"].to_numpy(dtype=float)
    remaining_opens = max_opens - labels["left"].to_numpy(dtype=float)
    height = labels["height"].to_numpy(dtype=float)
    forced = (height <= 0) | (remaining_opens <= 0) | (remaining_dyck <= height)
    forced_fraction = float(forced[target_mask].mean())
    free_fraction = 1.0 - forced_fraction
    return {
        "oracle_eval_dyck_acc_ceiling": forced_fraction + 0.5 * free_fraction,
        "forced_target_fraction": forced_fraction,
        "free_target_fraction": free_fraction,
    }


def compute_final_forced_free_behavior(summary: pd.DataFrame) -> pd.DataFrame:
    try:
        from scripts.task_a_extra_probes import (
            choose_indices,
            load_labels,
            model_predictions,
            prepare_labels,
            split_masks,
        )
    except Exception as exc:
        print(f"skip forced/free behavior split: {exc}")
        return pd.DataFrame()

    rows = []
    for _, row in summary.iterrows():
        run_dir = ROOT / str(row["run_dir"])
        config_path = run_dir / "config.json"
        labels_path = run_dir / "hidden_states" / "final" / "labels.parquet"
        if not config_path.exists() or not labels_path.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        task = config.get("task", {})
        label_row = SimpleNamespace(
            setting=str(row["experiment"]),
            run_dir_abs=run_dir,
            total_length=int(task["total_length"]),
            noise_vocab=int(task["num_noise_tokens"]),
        )
        labels = prepare_labels(load_labels(run_dir), label_row)
        target_mask = labels["has_target_label"].to_numpy(dtype=bool) & labels["target_is_dyck_position"].to_numpy(dtype=bool)
        eval_idx = choose_indices(target_mask, max_rows=200_000, seed=123)
        if len(eval_idx) == 0:
            continue
        target = labels.iloc[eval_idx]["target_token"].to_numpy(dtype=int)
        pred, _ = model_predictions(run_dir, int(task["num_noise_tokens"]), eval_idx)
        correct = pred == target
        eval_labels = labels.iloc[eval_idx].reset_index(drop=True)
        for split_name, split_mask in split_masks(eval_labels).items():
            if split_mask.sum() == 0:
                continue
            rows.append(
                {
                    "experiment": row["experiment"],
                    "run_key": row["run_key"],
                    "split": split_name,
                    "n": int(split_mask.sum()),
                    "fraction": float(split_mask.mean()),
                    "model_acc": float(correct[split_mask].mean()),
                    "oracle_acc": float(eval_labels.loc[split_mask, "oracle_next_dyck_acc"].mean()),
                    "gap_model_minus_oracle": float(
                        correct[split_mask].mean() - eval_labels.loc[split_mask, "oracle_next_dyck_acc"].mean()
                    ),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def final_forced_free_section(final_forced_free: pd.DataFrame) -> str:
    title = "## Final Forced/Free Behavior Split"
    if final_forced_free.empty:
        return (
            f"{title}\n\n"
            "final forced/free behavior split 还没有生成。需要 final labels 和模型 checkpoint 才能计算。"
        )
    forced = final_forced_free[final_forced_free["split"].eq("forced")]
    free = final_forced_free[final_forced_free["split"].eq("free")]
    all_targets = final_forced_free[final_forced_free["split"].eq("all_dyck_targets")]
    return (
        f"{title}\n\n"
        "**这张表是解释 raw accuracy 的关键。** `eval_dyck_acc` 把 forced 和 free 混在一起；"
        "forced 位置是规则已经唯一决定下一步，free 位置是 open/close 都合法、sampler 随机二选一。\n\n"
        f"结果上，final forced acc 范围是 `{forced['model_acc'].min():.3f}-{forced['model_acc'].max():.3f}`，"
        f"free acc 范围是 `{free['model_acc'].min():.3f}-{free['model_acc'].max():.3f}`，"
        f"overall raw Dyck acc 范围是 `{all_targets['model_acc'].min():.3f}-{all_targets['model_acc'].max():.3f}`。"
        "因此 Task B 和 Task A 是一致的：模型已经学会 forced 规则；raw accuracy 低主要是 free step 的随机性。\n\n"
        + markdown_table(final_forced_free, max_rows=20)
    )


def read_this_first(summary: pd.DataFrame) -> str:
    clean = summary[summary["experiment"].astype(str).str.contains("clean_short")]
    noisy = summary[summary["experiment"].astype(str).str.contains("noisy_short")]
    clean_row = clean.iloc[0] if not clean.empty else pd.Series(dtype=object)
    noisy_row = noisy.iloc[0] if not noisy.empty else pd.Series(dtype=object)
    return (
        "## 先读这个\n\n"
        "**这组实验要区分两个问题。** 第一，hidden state 里是否已经有一个可线性读出的 Dyck height/counting feature；"
        "第二，模型输出 open/close 时是否真的稳定使用了这个 feature。两件事不能用同一个指标替代。\n\n"
        "**当前最重要的结果是：**\n\n"
        f"- clean-short 的 count feature 很早出现：probe emergence step=`{format_value(clean_row.get('probe_emergence_step'))}`，"
        f"final height R2=`{format_value(clean_row.get('final_height_r2'))}`。\n"
        f"- noisy-short 的 count feature 明显更晚：probe emergence step=`{format_value(noisy_row.get('probe_emergence_step'))}`，"
        f"final height R2=`{format_value(noisy_row.get('final_height_r2'))}`。\n"
        "- `behavior_emergence_step` 现在按 `forced_acc >= 0.95` 判断："
        f"clean-short 在 step `{format_value(clean_row.get('behavior_emergence_step'))}` 过阈值，"
        f"noisy-short 在 step `{format_value(noisy_row.get('behavior_emergence_step'))}` 过阈值。\n"
        "- raw `eval_dyck_acc` 仍然不会到 0.8，但这不表示 forced 规则没有学会。当前 Dyck 生成器有大量 free choice，"
        f"clean 的 empirical oracle ceiling 约 `{format_value(clean_row.get('oracle_eval_dyck_acc_ceiling'))}`，"
        f"noisy 的 empirical oracle ceiling 约 `{format_value(noisy_row.get('oracle_eval_dyck_acc_ceiling'))}`。"
        "也就是说，按 raw next-token exact match 计算，0.8 不是当前数据分布下合理的成功线。\n"
        "- 后面的七个追加实验主要是在排除替代解释：这个 readout 是否只是位置/随机初始化，方向是否跨 checkpoint 稳定，"
        "output head 是否沿该方向读出，以及直接干预这个 scalar 是否足以改变行为。\n\n"
        "**一句话结论：** clean 数据下 height readout 很早变强；加入 noise 会推迟 readout。"
        "但目前证据更支持“count 是可读的表示”，还没有证明“同一个 probe direction 是直接控制 open/close 的因果旋钮”。"
    )


def metric_definitions() -> str:
    return (
        "## Metric Definitions\n\n"
        "这些指标分成三类：representation readout、behavior readout、以及二者的时间差。"
        "当前版本已经把 `behavior_emergence_step` 从 raw accuracy 阈值改成 forced accuracy 阈值；"
        "raw accuracy 只用来解释整体 next-token exact match 为什么停在 oracle ceiling 附近。\n\n"
        "| 指标 | 定义 | 本 notebook 中如何解读 |\n"
        "|---|---|\n"
        "| `height_r2` | 用某层 hidden state 线性回归当前 Dyck height 的 held-out R2。| 主要 representation 指标；高 R2 表示 count 可线性读出。|\n"
        "| `height_class_accuracy` | 把 height 当作离散 class 的 probe accuracy，用作 exact/rounded count readout 的近似。| 检查 readout 是否只是粗相关；越高说明 count 更接近离散可读。|\n"
        "| `legal_next_class_accuracy` | probe 能否读出下一步 open/close 是否都 legal。| 比 height 更接近 open/close 决策所需信息，但仍是 probe，不是行为本身。|\n"
        "| `eval_dyck_acc` | 模型 next-token prediction 在 Dyck token target 上的 raw accuracy。| 对当前 Markov/free-choice Dyck 不能直接用 0.8 当成功线；约 0.6 已接近当前数据的 raw 平台。|\n"
        "| `max_eval_dyck_acc` | 训练日志里所有 eval step 的最大 raw Dyck next-token accuracy。| 用来确认 raw exact-match 是否贴近 oracle ceiling，而不是判断 behavior emergence。|\n"
        "| `oracle_eval_dyck_acc_ceiling` | 根据 final labels 中 forced/free target 比例估计的 raw exact-match Bayes 上限：forced=1，free=0.5。| 判断 raw accuracy 的合理上限；当前 clean/noisy 都约 0.60。|\n"
        "| `forced_target_fraction` / `free_target_fraction` | Dyck target 中规则唯一决定下一步的比例 / open 和 close 都合法、sampler 随机二选一的比例。| free 占比高时，整体 raw accuracy 会被 0.5 随机上限拉低。|\n"
        "| `probe_emergence_step` | 第一个满足 `height_r2 >= 0.8` 且 `height_class_accuracy >= 0.5` 的 checkpoint。| 表示可读 count feature 第一次达到阈值。|\n"
        "| `behavior_metric` / `behavior_threshold` | 当前用于定义 behavior emergence 的指标和阈值。| 现在是 `forced_acc >= 0.95`。|\n"
        "| `behavior_emergence_step` | 第一个满足 `forced_acc >= 0.95` 的 checkpoint step。| 表示模型第一次基本学会 forced Dyck 规则。|\n"
        "| `verbalization_lag` | `behavior_emergence_step - probe_emergence_step`。| 正数表示 count probe 先出现；负数表示 forced 行为先于当前 probe 阈值出现。|"
    )


def comparison_table(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "experiment",
        "behavior_metric",
        "behavior_threshold",
        "probe_emergence_step",
        "behavior_emergence_step",
        "verbalization_lag",
        "early_feature_layer",
        "best_feature_layer",
        "final_height_r2",
        "final_height_class_accuracy",
        "final_legal_next_class_accuracy",
        "max_eval_dyck_acc",
        "oracle_eval_dyck_acc_ceiling",
        "forced_target_fraction",
        "free_target_fraction",
        "final_forced_acc",
        "final_free_acc",
        "final_oracle_acc",
        "final_gap_model_minus_oracle",
        "final_eval_dyck_acc",
    ]
    return summary[[column for column in columns if column in summary.columns]]


def markdown_table(frame: pd.DataFrame, *, max_rows: int) -> str:
    view = frame.head(max_rows).copy()
    headers = [str(column) for column in view.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(format_value(row[column]) for column in view.columns) + " |")
    if len(frame) > max_rows:
        lines.append(f"\n只显示前 {max_rows} 行；完整表在 `results/dyck_counter_task_b_training_dynamics/`。")
    return "\n".join(lines)


def format_value(value) -> str:
    if value is None:
        return "NA"
    try:
        if pd.isna(value):
            return "NA"
    except TypeError:
        pass
    if isinstance(value, float):
        if math.isfinite(value) and abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:.3f}"
    return str(value)


def value_range(df: pd.DataFrame, column: str) -> str:
    if df.empty or column not in df:
        return "NA"
    values = df[column].dropna().astype(float)
    if values.empty:
        return "NA"
    if abs(values.min() - values.max()) < 1e-9:
        return f"{values.iloc[0]:.3f}"
    return f"{values.min():.3f}-{values.max():.3f}"


def order_summary(summary: pd.DataFrame) -> pd.DataFrame:
    order = {name: index for index, name in enumerate(DEFAULT_EXPERIMENT_ORDER)}
    summary = summary.copy()
    summary["_order"] = summary["experiment"].map(lambda value: order.get(str(value), len(order)))
    return summary.sort_values(["_order", "experiment", "run_key"]).drop(columns=["_order"]).reset_index(drop=True)


def load_run_config(summary_row: pd.Series) -> dict:
    path = ROOT / str(summary_row["run_dir"]) / "config.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def setting_title(config: dict, summary_row: pd.Series) -> str:
    name = str(summary_row["experiment"]).removeprefix("dyck_counter_task_b_").removesuffix("_smoke")
    return name.replace("_", "-")


def setting_description(config: dict, summary_row: pd.Series) -> str:
    task = config.get("task", {})
    checkpoint_steps = config.get("checkpoint_steps", [])
    training_steps = config.get("training_steps", "NA")
    batch_size = config.get("batch_size", "NA")
    lr = config.get("learning_rate", "NA")
    return (
        f"- run: `{summary_row['run_key']}`。\n"
        f"- task: `dyck_pairs={task.get('dyck_pairs')}`，`total_length={task.get('total_length')}`，"
        f"`seq_len={task.get('seq_len')}`，`repeat_prob={task.get('repeat_prob')}`，"
        f"`num_noise_tokens={task.get('num_noise_tokens')}`。\n"
        f"- model: `{config.get('model_name')}`，3-layer decoder-only Transformer，seed=`{config.get('seed')}`。\n"
        f"- training: `{training_steps}` steps，batch_size=`{batch_size}`，lr=`{lr}`。\n"
        f"- checkpoints: `{checkpoint_steps}` plus `final`。\n"
        "- extraction/probes: all layers / all positions；ridge targets 为 `left/right/height`，classification targets 为 "
        "`height_class/left_right_class/legal_next_class`。"
    )


def interpretation(row: pd.Series, best_run: pd.DataFrame, config: dict) -> str:
    probe_step = format_value(row.get("probe_emergence_step"))
    behavior_step = format_value(row.get("behavior_emergence_step"))
    behavior_metric = format_value(row.get("behavior_metric"))
    behavior_threshold = format_value(row.get("behavior_threshold"))
    lag = format_value(row.get("verbalization_lag"))
    early_layer = format_value(row.get("early_feature_layer"))
    best_layer = format_value(row.get("best_feature_layer"))
    final_r2 = format_value(row.get("final_height_r2"))
    final_count = format_value(row.get("final_height_class_accuracy"))
    final_legal = format_value(row.get("final_legal_next_class_accuracy"))
    final_behavior = format_value(row.get("final_eval_dyck_acc"))
    max_behavior = format_value(row.get("max_eval_dyck_acc"))
    oracle_behavior = format_value(row.get("oracle_eval_dyck_acc_ceiling"))
    forced_fraction = format_value(row.get("forced_target_fraction"))
    free_fraction = format_value(row.get("free_target_fraction"))
    final_forced = format_value(row.get("final_forced_acc"))
    final_free = format_value(row.get("final_free_acc"))
    final_gap = format_value(row.get("final_gap_model_minus_oracle"))
    initial = best_run.sort_values("checkpoint_order").iloc[0]
    initial_r2 = format_value(initial.get("height_r2"))
    initial_count = format_value(initial.get("height_class_accuracy"))
    seq_len = config.get("task", {}).get("seq_len")
    repeat_prob = config.get("task", {}).get("repeat_prob")
    has_noise = repeat_prob is not None and float(repeat_prob) < 1.0
    setting_point = (
        "这里测试的是：把 Dyck token 嵌入到含 noise token 的序列后，count feature 的形成是否会被推迟。"
        if has_noise
        else "这里测试的是：在没有额外 noise token 的短序列里，count feature 会多早变成线性可读。"
    )

    return (
        f"**目的。** {setting_point}\n\n"
        "**结果。**\n\n"
        f"- count readout 第一次过阈值在 step `{probe_step}`，最早可读层是 layer `{early_layer}`。\n"
        f"- final checkpoint 的 best layer 是 layer `{best_layer}`：height R2=`{final_r2}`，"
        f"height-class acc=`{final_count}`，legal-next acc=`{final_legal}`。\n"
        f"- behavior emergence 用 `{behavior_metric} >= {behavior_threshold}` 判断，在 step `{behavior_step}` 过阈值；"
        f"verbalization lag=`{lag}`。\n"
        f"- final forced acc=`{final_forced}`，free acc=`{final_free}`，model-oracle gap=`{final_gap}`。\n"
        f"- raw Dyck next-token accuracy 最大值是 `{max_behavior}`，最后是 `{final_behavior}`；"
        f"empirical oracle ceiling 约 `{oracle_behavior}`，其中 forced fraction=`{forced_fraction}`，free fraction=`{free_fraction}`。\n"
        f"- step 0 已有 baseline：height R2=`{initial_r2}`，height-class acc=`{initial_count}`。\n\n"
        "**说明。**\n\n"
        f"- step 0 baseline 不能忽略。`seq_len={seq_len}`、`repeat_prob={repeat_prob}` 的生成方式会让 position/token embedding 本身带有部分 height 信息，"
        "所以不能只看“随机模型也能 probe 到一点”。真正有意义的是训练后 readout 是否超过这些 baseline、是否更稳定。\n"
        "- raw Dyck accuracy 在当前 stochastic/free-choice 生成器下有约 0.6 的 oracle ceiling，"
        "所以 behavior emergence 应看 forced acc 或 oracle-normalized gap。\n"
        "- 若 verbalization lag 为负，含义不是“probe 错了”，而是当前 high-R2 height probe 阈值比 forced-rule 行为更严格；"
        "模型可能先学会边界规则，再逐渐形成更完整的线性 height readout。"
    )


def followup_notes(row: pd.Series, config: dict) -> str:
    behavior_step = row.get("behavior_emergence_step")
    final_behavior = row.get("final_eval_dyck_acc")
    task = config.get("task", {})
    notes = [
        "当前已经用 forced acc 定义 behavior emergence；下一步应补 oracle-normalized gap curve，检查 overall 是否贴近 Bayes ceiling。",
        "补 multi-seed，确认 clean/noisy 的 probe emergence step 差异不是 seed=0 的偶然结果。",
    ]
    if pd.isna(behavior_step):
        notes.append(
            f"当前 `seq_len={task.get('seq_len')}` 下 forced acc 未过阈值；需要检查 forced split 的错误集中在 must-open 还是 must-close。"
        )
    else:
        notes.append("behavior 已过 forced 阈值；重点看它相对 probe emergence 是先出现还是后出现。")
    if not pd.isna(final_behavior) and float(final_behavior) < 0.7:
        notes.append("raw exact-match 受 oracle ceiling 限制；后续更适合用 layer-wise activation patch 验证 forced 规则行为是否依赖 counter readout。")
    return "\n".join(f"- {note}" for note in notes)


def extension_cells(output: Path) -> list[dict]:
    if not EXT_DIR.exists():
        return [
            md(
                "## 3. Additional Training Dynamics Experiments\n\n"
                "扩展实验结果还没有生成。运行 `scripts/task_b_training_dynamics_extensions.py` 后，本节会自动填充七个追加实验。"
            )
        ]

    tables = {
        "extended": read_optional_csv("experiment_1_extended_training.csv"),
        "controls": read_optional_csv("experiment_2_position_random_controls.csv"),
        "transfer": read_optional_csv("experiment_3_probe_transfer.csv"),
        "alignment": read_optional_csv("experiment_4_output_head_alignment.csv"),
        "branch": read_optional_csv("experiment_4b_branch_manifold_split.csv"),
        "intervention": read_optional_csv("experiment_5_checkpoint_intervention.csv"),
        "length": read_optional_csv("experiment_6_length_generalization.csv"),
        "density": read_optional_csv("experiment_7_density_curve.csv"),
    }

    cells = [
        md(
            "## 3. Additional Training Dynamics Experiments\n\n"
            "前两节说明了 clean/noisy 的 count readout 何时出现，但还留下几个歧义。"
            "下面七个实验分别对应七个具体问题：\n\n"
            "1. 训练更久能不能让 raw behavior 追上 probe？\n"
            "2. height probe 是否只是 position 或随机初始化造成的假象？\n"
            "3. 不同 checkpoint 的 probe direction 是不是同一个稳定方向？\n"
            "4. output head 是否直接沿 height direction 读出 open/close？\n"
            "5. 直接移动 height scalar 是否能改变模型行为？\n"
            "6. 训练中的行为改善能不能泛化到更长/更稀疏的评估长度？\n"
            "7. Task A 里看到的失败是否主要来自 bracket supervision density 太低？"
        )
    ]
    cells.append(
        extension_section(
            "3.1 Extended Training",
            tables["extended"],
            figure="experiment_1_extended_training.png",
            output=output,
            intro=extended_training_text(tables["extended"]),
            max_rows=10,
        )
    )
    cells.append(
        extension_section(
            "3.2 Position/Random Baseline Controls",
            tables["controls"],
            figure="experiment_2_position_random_controls.png",
            output=output,
            intro=baseline_controls_text(tables["controls"]),
            max_rows=12,
        )
    )
    cells.append(
        extension_section(
            "3.3 Checkpoint Probe Transfer",
            summarize_transfer(tables["transfer"]),
            figure="experiment_3_probe_transfer.png",
            output=output,
            intro=probe_transfer_text(tables["transfer"]),
            max_rows=20,
        )
    )
    cells.append(
        extension_section(
            "3.4 Output-Head Alignment Over Time",
            summarize_alignment(tables["alignment"]),
            figure="experiment_4_output_head_alignment.png",
            output=output,
            intro=output_alignment_text(tables["alignment"]),
            max_rows=20,
        )
    )
    cells.append(
        extension_section(
            "3.4b Open/Close Branch Manifold Split",
            summarize_branch_manifold(tables["branch"]),
            figure="experiment_4b_branch_manifold_split.png",
            output=output,
            intro=branch_manifold_text(tables["branch"]),
            max_rows=12,
        )
    )
    cells.append(
        extension_section(
            "3.5 Checkpoint-Wise Direct Intervention",
            summarize_intervention(tables["intervention"]),
            figure="experiment_5_checkpoint_intervention.png",
            output=output,
            intro=intervention_text(tables["intervention"]),
            max_rows=20,
        )
    )
    cells.append(
        extension_section(
            "3.6 Length Generalization Over Training",
            tables["length"],
            figure="experiment_6_length_generalization.png",
            output=output,
            intro=length_generalization_text(tables["length"]),
            max_rows=24,
        )
    )
    cells.append(
        extension_section(
            "3.7 Bracket-Density Curve",
            tables["density"],
            figure="experiment_7_density_curve.png",
            output=output,
            intro=density_curve_text(tables["density"]),
            max_rows=20,
        )
    )
    return cells


def read_optional_csv(name: str) -> pd.DataFrame:
    path = EXT_DIR / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def extension_section(
    title: str,
    table: pd.DataFrame,
    *,
    figure: str,
    output: Path,
    intro: str,
    max_rows: int,
) -> dict:
    fig_path = EXT_FIG_DIR / figure
    figure_md = f"\n\n![{title}]({rel_from_notebook(output, fig_path)})" if fig_path.exists() else ""
    table_md = markdown_table(table, max_rows=max_rows) if not table.empty else "结果表还没有生成。"
    return md(f"### {title}\n\n{intro}\n\n{table_md}{figure_md}")


def extended_training_text(df: pd.DataFrame) -> str:
    if df.empty:
        return "**问题。** 训练更久以后，raw behavior 会不会追上 probe？"
    view = df[["experiment", "training_steps", "final_eval_dyck_acc", "final_height_r2"]].copy()
    base = view[view["training_steps"].eq(2000)]
    extended = view[view["training_steps"].eq(5000)]
    base_acc = f"{base['final_eval_dyck_acc'].min():.3f}-{base['final_eval_dyck_acc'].max():.3f}" if not base.empty else "NA"
    ext_acc = (
        f"{extended['final_eval_dyck_acc'].min():.3f}-{extended['final_eval_dyck_acc'].max():.3f}"
        if not extended.empty
        else "NA"
    )
    ext_r2 = f"{extended['final_height_r2'].min():.3f}-{extended['final_height_r2'].max():.3f}" if not extended.empty else "NA"
    return (
        "**问题。** 训练更久以后，raw behavior 会不会追上 probe？\n\n"
        "**做法。** 额外训练 clean/noisy 的 5k-step 版本，并和原来的 2k-step smoke run 对比。\n\n"
        f"**结果。** 2k final raw Dyck accuracy 在 `{base_acc}`，5k final raw Dyck accuracy 在 `{ext_acc}`；"
        f"5k final height R2 在 `{ext_r2}`。\n\n"
        "**解释。** 训练更久没有把 raw Dyck accuracy 推到 0.8，说明 0.8 raw threshold 不是合适的行为成功线。"
        "同时 height probe 仍然很强，说明 representation readout 和 raw next-token behavior 需要分开报告。"
    )


def baseline_controls_text(df: pd.DataFrame) -> str:
    if df.empty:
        return "**问题。** height probe 是否只是 position 或随机初始化带来的假象？"
    trained = df[df["control"].eq("trained_hidden_final")]["height_r2"]
    shuffled = df[df["control"].eq("trained_hidden_shuffled_height")]["height_r2"]
    random = df[df["control"].eq("random_model_hidden_step0")]["height_r2"]
    pos = df[df["control"].eq("position_one_hot")]["height_r2"]
    return (
        "**问题。** height probe 是否只是 position 或随机初始化带来的假象？\n\n"
        "**做法。** 比较五种输入给同一类 ridge probe，目标都是预测同一批 token position 上的 true height：\n\n"
        "- `position_one_hot`：只给 probe 绝对位置 one-hot，不给任何模型 hidden state。它检验 height 是否仅由序列位置分布就能猜出来。\n"
        "- `position_plus_progress`：只给两个手工标量特征：normalized absolute position 和 normalized `dyck_seen`。"
        "它检验 height 是否能由位置加 Dyck 进度这种低维 schedule 信息解释。\n"
        "- `random_model_hidden_step0`：给 step 0 随机初始化模型的 hidden state，使用和 final 模型相同的层。"
        "它检验随机 embedding/position encoding/未训练 Transformer 是否已经线性携带 height 信息。\n"
        "- `trained_hidden_final`：给训练完成后的 final hidden state。这是我们真正关心的模型表示。\n"
        "- `trained_hidden_shuffled_height`：仍给 final hidden state，但把 height label 随机打乱。"
        "这是负控；如果它也高，说明 probe 或切分流程有问题。\n\n"
        f"**结果。** trained final hidden R2=`{trained.min():.3f}-{trained.max():.3f}`；"
        f"random step0 hidden R2=`{random.min():.3f}-{random.max():.3f}`；"
        f"position-only R2≈`{pos.mean():.3f}`；shuffled-label R2≈`{shuffled.mean():.3f}`。\n\n"
        "**解释。** position 和随机初始化确实贡献了一部分可读信息，尤其 clean setting 的 step0 baseline 较高。"
        "但 trained hidden 明显强于这些 baseline，shuffled label 接近 0，说明最终 readout 不是纯伪相关。"
    )


def summarize_transfer(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    rows = []
    for experiment, group in df.groupby("experiment", sort=False):
        same = group[group["source_step"].eq(group["target_step"])]
        early_to_final = group[
            group["source_step"].eq(group["source_step"].min()) & group["target_step"].eq(group["target_step"].max())
        ]
        final_to_early = group[
            group["source_step"].eq(group["source_step"].max()) & group["target_step"].eq(group["target_step"].min())
        ]
        rows.append(
            {
                "experiment": experiment,
                "mean_self_transfer_r2": same["transfer_r2"].mean(),
                "step0_to_final_r2": early_to_final["transfer_r2"].mean(),
                "final_to_step0_r2": final_to_early["transfer_r2"].mean(),
            }
        )
    return pd.DataFrame(rows)


def probe_transfer_text(df: pd.DataFrame) -> str:
    if df.empty:
        return "**问题。** 不同 checkpoint 的 height probe direction 是不是同一个稳定方向？"
    same = df[df["source_step"].eq(df["target_step"])]["transfer_r2"].mean()
    off = df[~df["source_step"].eq(df["target_step"])]["transfer_r2"].mean()
    return (
        "**问题。** 不同 checkpoint 的 height probe direction 是不是同一个稳定方向？\n\n"
        "**做法。** 在 checkpoint i 训练 height probe，然后冻结 probe 到 checkpoint j 上测试；"
        "如果方向稳定，off-checkpoint transfer R2 应该仍然较高。\n\n"
        f"**结果。** self-checkpoint 平均 R2≈`{same:.3f}`，off-checkpoint 平均 R2≈`{off:.3f}`，"
        "很多早晚 checkpoint 互测为负。\n\n"
        "**解释。** count information 可以线性读出，不代表 readout 坐标从训练一开始就稳定。"
        "尤其 step0 的 high probe 不能直接当作最终机制；训练过程中表示坐标发生了明显重排。"
    )


def summarize_alignment(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    rows = []
    for _, group in df.groupby(["experiment", "checkpoint"], sort=False):
        idx = group["height_r2"].astype(float).idxmax()
        rows.append(group.loc[idx])
    columns = [
        "experiment",
        "checkpoint",
        "step",
        "layer",
        "height_r2",
        "cosine_height_dir_close_minus_open",
        "corr_height_axis_with_close_minus_open_margin",
    ]
    return pd.DataFrame(rows)[columns]


def output_alignment_text(df: pd.DataFrame) -> str:
    if df.empty:
        return "**问题。** output head 是否直接沿 height direction 读出 open/close？"
    best = summarize_alignment(df)
    cos = best["cosine_height_dir_close_minus_open"].mean()
    corr = best["corr_height_axis_with_close_minus_open_margin"].mean()
    return (
        "**问题。** output head 是否直接沿 height direction 读出 open/close？\n\n"
        "**做法。** 对每个 checkpoint 选择 height R2 最好的层，比较 probe direction 和 output head 的 "
        "`close_minus_open` 方向：一个是几何 cosine，一个是该 axis projection 与 logit margin 的相关。\n\n"
        f"**结果。** best-layer 平均 cosine≈`{cos:.3f}`，axis-margin correlation≈`{corr:.3f}`。\n\n"
        "**解释。** output head 没有简单地把 probe 向量当作 close/open 权重方向。"
        "但是在数据流形上，height-axis projection 和 close-minus-open margin 有正相关，说明 count 信息可能通过更复杂的表示几何进入输出。"
    )


def summarize_branch_manifold(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    final = df[df["checkpoint"].eq("final")].copy()
    if final.empty:
        final = df.loc[df.groupby(["experiment", "branch_source"])["step"].idxmax()].copy()
    columns = [
        "experiment",
        "branch_source",
        "step",
        "layer",
        "height_r2",
        "branch_acc_full_hidden",
        "branch_acc_height_axis_only",
        "branch_acc_residual_after_height",
        "height_dir_close_open_cosine",
        "cosine_residual_branch_dir_close_minus_open_head",
        "corr_residual_branch_axis_with_branch",
    ]
    return final[[column for column in columns if column in final.columns]].sort_values(["experiment", "branch_source"])


def branch_manifold_text(df: pd.DataFrame) -> str:
    if df.empty:
        return "**问题。** hidden state 是否能分出 open/close 两条流形，并且它们是否共用同一个 count direction？"
    final = summarize_branch_manifold(df)
    current = final[final["branch_source"].eq("current_token")]
    next_target = final[final["branch_source"].eq("next_target")]
    current_res = value_range(current, "branch_acc_residual_after_height")
    current_height_only = value_range(current, "branch_acc_height_axis_only")
    current_height_cos = value_range(current, "height_dir_close_open_cosine")
    next_res = value_range(next_target, "branch_acc_residual_after_height")
    next_height_cos = value_range(next_target, "height_dir_close_open_cosine")
    return (
        "**问题。** hidden state 是否能分出 open/close 两条流形，并且这两条流形是否共用同一个 count direction？\n\n"
        "**做法。** 对每个 checkpoint 的 best height layer 做两个版本的 branch label：\n\n"
        "- `current_token`：当前位置已经看到的 bracket 是 open 还是 close。这个对应我们在 hidden-state 图里看到的两条 open/close 曲线。\n"
        "- `next_target`：当前位置要预测的下一个 bracket target 是 open 还是 close。这个更接近 output head 的 next-token 决策，但 free step 本身有随机性。\n\n"
        "每个版本都做三件事：先拟合 height/count direction；再比较只用 height axis 能否分类 branch；"
        "最后把 height direction 从 hidden state 投影掉，看 residual subspace 里还能不能线性分开 open/close。"
        "另外分别在 open/close 子集上拟合 height direction，计算两者 cosine，检验 count direction 是否共享。\n\n"
        f"**结果。** final checkpoint 上，`current_token` 在去掉 height direction 后的 residual branch accuracy 是 `{current_res}`，"
        f"而只用 height axis 是 `{current_height_only}`；这说明当前 token 的 open/close 两条流形几乎完全可分，而且分离方向不只是 count axis。"
        f"同一批 `current_token` 分支各自拟合出的 height direction cosine 是 `{current_height_cos}`，属于中等对齐；"
        f"`next_target` 的 residual branch accuracy 只有 `{next_res}`，但 next-open/next-close 子集的 height direction cosine 更高，为 `{next_height_cos}`。\n\n"
        "**解释。** 更准确的结论是：hidden state 里存在一个强的全局 count direction，同时 residual subspace 里还有当前 token open/close 的 branch direction。"
        "这能解释我们看到的两条曲线。"
        "但“两个分支各自的 height direction 完全相同”并不是严格成立，current-token 分支的单独拟合方向只有中等一致；"
        "可能是 current token embedding offset 和 height/parity 结构共同影响。"
        "`next_target` 不会形成同样干净的两条流形，因为 free step 的下一个 open/close 是随机采样；"
        "因此 output head alignment 不能只看 height direction，也不能期待 next-target branch 像 current-token branch 一样完全可分。"
    )


def summarize_intervention(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    rows = []
    focused = df[df["split"].eq("all_dyck_targets")]
    for keys, group in focused.groupby(["experiment", "checkpoint", "step", "layer"], sort=False):
        x = group["delta_height_axis_std"].to_numpy(dtype=float)
        rows.append(
            {
                "experiment": keys[0],
                "checkpoint": keys[1],
                "step": keys[2],
                "layer": keys[3],
                "p_close_slope": np.polyfit(x, group["mean_p_close_given_bracket"].to_numpy(dtype=float), deg=1)[0],
                "accuracy_slope": np.polyfit(x, group["accuracy"].to_numpy(dtype=float), deg=1)[0],
            }
        )
    return pd.DataFrame(rows)


def intervention_text(df: pd.DataFrame) -> str:
    if df.empty:
        return "**问题。** 直接移动 height scalar 是否能改变模型行为？"
    slopes = summarize_intervention(df)
    return (
        "**问题。** 直接移动 height scalar 是否能改变模型行为？\n\n"
        "**做法。** 在每个 checkpoint 的 best layer 上，沿 height probe direction 加减若干个 axis standard deviation，"
        "再看 bracket target 上的 P(close) 和 accuracy 斜率。\n\n"
        f"**结果。** 平均 P(close) slope≈`{slopes['p_close_slope'].mean():.4f}`，"
        f"accuracy slope≈`{slopes['accuracy_slope'].mean():.4f}`。\n\n"
        "**解释。** 这个 direct hidden intervention 的效应很小，支持“probe scalar 可读但不是强因果旋钮”。"
        "注意这不是完整 forward activation patch；真正的因果验证还需要在中间层 patch 后继续跑后续层。"
    )


def length_generalization_text(df: pd.DataFrame) -> str:
    if df.empty:
        return "**问题。** 训练中的行为改善能否泛化到更长/更稀疏的评估长度？"
    final = df[df["checkpoint"].eq("final")]
    if final.empty:
        final = df[df["step"].eq(df["step"].max())]
    short = final[final["eval_seq_len"].eq(final["eval_seq_len"].min())]
    long = final[final["eval_seq_len"].eq(final["eval_seq_len"].max())]
    return (
        "**问题。** 训练中的行为改善能否泛化到更长/更稀疏的评估长度？\n\n"
        "**做法。** 固定 checkpoint，额外评估多个 `eval_seq_len` 和 `repeat_prob`，让 Dyck token 在更长上下文中更稀疏。\n\n"
        f"**结果。** final checkpoint 的 Dyck accuracy 范围是 `{final['dyck_accuracy'].min():.3f}-{final['dyck_accuracy'].max():.3f}`；"
        f"最短 eval 长度均值≈`{short['dyck_accuracy'].mean():.3f}`，最长 eval 长度均值≈`{long['dyck_accuracy'].mean():.3f}`。\n\n"
        "**解释。** 更长/更稀疏的评估会压低行为，说明短分布上的训练改善不等价于稳健长度泛化。"
        "这和 Task A 的 density 结论一致。"
    )


def density_curve_text(df: pd.DataFrame) -> str:
    if df.empty:
        return "**问题。** Task A/Task B 里的 long sparse 失败，主要来自长上下文本身，还是来自 bracket supervision 太稀疏？"
    transition = df[df.get("forced_accuracy", pd.Series(dtype=float)) >= 0.8]
    first = int(transition["bracket_tokens"].min()) if not transition.empty else "NA"
    free_transition = df[df.get("free_accuracy", pd.Series(dtype=float)) >= 0.45]
    first_free = int(free_transition["bracket_tokens"].min()) if not free_transition.empty else "NA"
    return (
        "**问题。** Task A/Task B 里的 long sparse 失败，主要来自长上下文本身，还是来自 bracket supervision 太稀疏？\n\n"
        "**做法。** 固定 `seq_len=2000`，只扫 bracket token 数量；这样长度不变，只改变 Dyck supervision density。\n\n"
        f"**结果。** forced accuracy 第一次超过 0.8 约在 `{first}` 个 bracket token；"
        f"free accuracy 接近 0.5 平台约在 `{first_free}` 个 bracket token。"
        "height R2 在很低 density 时也能保持可读，但 behavior 到 48-64 bracket tokens 后才明显恢复。\n\n"
        "**解释。** 失败的主要来源不是单纯 `seq_len=2000`，而是 Dyck target 在 next-token loss 中过于稀疏。"
        "这解释了为什么 tiny_extreme_long 可以有较高 height probe，却没有稳定 behavior。"
    )


def overall_interpretation() -> str:
    return (
        "## 4. Overall Interpretation\n\n"
        "**目前这批结果支持的结论：**\n\n"
        "1. Transformer hidden state 中确实会形成可线性读出的 Dyck height/counting feature。"
        "clean-short 很早出现，noisy-short 明显延后。\n"
        "2. forced behavior emergence 已经可以用 `forced_acc >= 0.95` 定义。"
        "当前 clean 在 step 50 过阈值，noisy 在 step 200 过阈值；"
        "raw Dyck next-token accuracy 仍只适合和 oracle ceiling 比较。\n"
        "3. position 和随机初始化能解释一部分早期 readout，尤其 clean setting；但 trained hidden 明显更强，"
        "shuffled-label 控制接近 0，所以最终 probe 不是纯假象。\n"
        "4. noisy setting 里 forced behavior 先于 high-R2 height probe 出现，说明边界规则行为和完整线性 height readout 不是同一个时间点。"
        "probe direction 在训练中并不总是稳定，output head 也没有简单对齐这个方向。"
        "因此“height 可线性读出”不能直接推出“这个线性方向就是模型的因果控制变量”。\n"
        "5. 长上下文失败更像是 bracket supervision density 问题：固定 `seq_len=2000` 时，"
        "bracket token 从 20 增加到 48-64 后 forced/free behavior 才明显恢复。\n\n"
        "**下一步最值得补的是：**\n\n"
        "- 把 oracle-normalized gap 也做成 checkpoint curve，和 forced behavior emergence 一起报告。\n"
        "- 做 multi-seed，确认 clean/noisy emergence timing 和 density threshold 是否稳定。\n"
        "- 做真正的 layer-wise activation patch：在中间层 patch donor activation 后继续 forward，"
        "而不是只在 final hidden 上沿 probe direction 做 direct-logit intervention。"
    )


def rel_from_notebook(notebook_path: Path, asset_path: Path) -> str:
    return os.path.relpath(asset_path.resolve(), start=notebook_path.resolve().parent)


if __name__ == "__main__":
    main()
