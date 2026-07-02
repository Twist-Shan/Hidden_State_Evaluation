from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default=str(ROOT / "results" / "needle_count_task_c_direct_count"))
    parser.add_argument("--figure-dir", default=str(ROOT / "figures" / "needle_count_task_c_direct_count"))
    parser.add_argument("--output", default=str(ROOT / "notebooks" / "Dyck_Syn_to_Rea" / "Task_C_Direct_Count_Readout.ipynb"))
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    figure_dir = Path(args.figure_dir)
    output = Path(args.output)

    config = json.loads((result_dir / "config.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(result_dir / "summary.csv")
    checkpoint = pd.read_csv(result_dir / "checkpoint_metrics.csv")
    probe = pd.read_csv(result_dir / "probe_readout.csv")
    steering_path = result_dir / "steering.csv"
    steering = pd.read_csv(steering_path) if steering_path.exists() else pd.DataFrame()

    cells = [
        md(
            "# Task C: From Hidden Count Feature to Direct Count Output\n\n"
            "这个 notebook 对应核心实验 C：模型是否先形成 internal counting feature，随后再把这个 feature "
            "转换成最终输出的 `NUM_k` answer token。\n\n"
            "当前版本是一个最小可运行 smoke test：用 `NeedleCount-synthetic` 做 single-token direct count。"
            "每条序列里随机放入 `k` 个 NEEDLE token，`k` 在 `0..max_count` 上均匀采样；末尾追加 query token，"
            "模型只在 query 位置预测一个答案 token `NUM_k`。"
        ),
        md(read_this_first(config, summary)),
        code(
            "from pathlib import Path\n"
            "import pandas as pd\n\n"
            "ROOT = Path.cwd()\n"
            "while ROOT != ROOT.parent and not (ROOT / 'results').exists():\n"
            "    ROOT = ROOT.parent\n"
            "RESULT_DIR = ROOT / 'results' / 'needle_count_task_c_direct_count'\n"
            "FIGURE_DIR = ROOT / 'figures' / 'needle_count_task_c_direct_count'\n"
            "summary = pd.read_csv(RESULT_DIR / 'summary.csv')\n"
            "checkpoint = pd.read_csv(RESULT_DIR / 'checkpoint_metrics.csv')\n"
            "probe = pd.read_csv(RESULT_DIR / 'probe_readout.csv')\n"
            "steering = pd.read_csv(RESULT_DIR / 'steering.csv')\n"
            "summary"
        ),
        md("## 1. 实验问题\n\n" + experiment_question()),
        md("## 2. Task 与训练设置\n\n" + setting_table(config)),
        md("## 3. 指标定义\n\n" + metric_definitions()),
        md("## 4. Main Result\n\n" + main_result(summary, checkpoint, probe)),
        md(
            "![Task C training dynamics]("
            + rel_from_notebook(output, figure_dir / "task_c_training_dynamics.png")
            + ")"
        ),
        md("### 4.1 Checkpoint-level behavior\n\n" + markdown_table(clean_checkpoint_table(checkpoint), max_rows=30)),
        md("### 4.2 Final layer-wise readout\n\n" + final_layerwise_section(probe)),
        md(
            "![Task C layer-wise readout]("
            + rel_from_notebook(output, figure_dir / "task_c_layerwise_readout.png")
            + ")"
        ),
        md("## 5. Final Query-Position Steering\n\n" + steering_section(steering)),
    ]
    if not steering.empty:
        cells.append(
            md(
                "![Task C final query steering]("
                + rel_from_notebook(output, figure_dir / "task_c_final_query_steering.png")
                + ")"
            )
        )
        cells.append(md("### 5.1 Steering table\n\n" + markdown_table(clean_steering_table(steering), max_rows=30)))

    cells.extend(
        [
            md("## 6. 当前结论\n\n" + interpretation(summary, checkpoint, probe, steering)),
            md("## 7. 下一步\n\n" + next_steps()),
            code(
                "# 可选：重新跑这个 smoke test 并重写 notebook\n"
                "# !python scripts/task_c_direct_count_readout.py --steps 1000 --checkpoint-steps 0,50,100,200,500,1000\n"
                "# !python scripts/write_task_c_direct_count_notebook.py"
            ),
        ]
    )

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {output}")


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


def read_this_first(config: dict, summary: pd.DataFrame) -> str:
    row = summary.iloc[0]
    return (
        "## Read This First\n\n"
        f"- 这次只跑了一个 smoke setting：`seq_len={config['seq_len']}`, `max_count={config['max_count']}`, "
        f"`noise_vocab={config['noise_vocab']}`, `steps={config['steps']}`, `seed={config['seed']}`。\n"
        f"- 末轮 answer accuracy 是 `{row['answer_acc']:.3f}`，best-count-layer R2 是 `{row['best_count_r2']:.3f}`，"
        f"best logit-lens accuracy 是 `{row['best_logit_lens_acc']:.3f}`。\n"
        "- 因为这是单 setting smoke test，结论只用于确认实验 C pipeline 能跑通；正式结论还需要扫 "
        "`context length / max count / distractor ratio / seed`。"
    )


def experiment_question() -> str:
    return (
        "Task A/B 已经说明 hidden state 里可以出现可线性读出的 counting feature。"
        "实验 C 问的是下一步：这个 feature 是否真的接到输出头，能否直接变成答案 token。\n\n"
        "因此这里不只看 `hidden -> count` probe，而是同时看四件事：\n\n"
        "1. 模型最终能否输出正确的 `NUM_k`。\n"
        "2. hidden count probe 是否已经可读。\n"
        "3. 同一个 hidden 直接过 final output head 时，logit lens 是否已经能读出 `NUM_k`。\n"
        "4. ridge count direction 是否和 `NUM_k -> NUM_{k+1}` 的 unembedding 方向对齐，并且能否被 steering 系统性推动。"
    )


def setting_table(config: dict) -> str:
    rows = pd.DataFrame(
        [
            {"item": "source task", "value": config["task"]},
            {"item": "sequence format", "value": config["format"]},
            {"item": "seq_len", "value": config["seq_len"]},
            {"item": "max_count", "value": config["max_count"]},
            {"item": "answer tokens", "value": f"NUM_0..NUM_{config['max_count']}"},
            {"item": "noise/distractor vocab", "value": config["noise_vocab"]},
            {"item": "model", "value": f"{config['n_layers']} layers, d_model={config['d_model']}, heads={config['n_heads']}"},
            {"item": "training", "value": f"{config['steps']} steps, batch={config['batch_size']}, lr={config['lr']}"},
            {"item": "checkpoints", "value": str(config["checkpoint_steps"])},
        ]
    )
    return markdown_table(rows, max_rows=30)


def metric_definitions() -> str:
    rows = pd.DataFrame(
        [
            {
                "metric": "answer_acc",
                "definition": "full-vocab top-1 是否等于真实答案 token `NUM_k`。",
                "interpretation": "真正的 direct-count 行为准确率。",
            },
            {
                "metric": "num_restricted_acc",
                "definition": "只在 `NUM_0..NUM_K` answer token 之间取 argmax 后是否等于真实 k。",
                "interpretation": "排除输出非答案 token 后，答案排序本身是否正确。",
            },
            {
                "metric": "count_r2",
                "definition": "在某层 query-position hidden 上训练 ridge regression 预测真实 count 的 test R2。",
                "interpretation": "hidden 中 count 是否线性可读。",
            },
            {
                "metric": "ridge_round_acc",
                "definition": "ridge 预测 count 后四舍五入并裁剪到合法范围的准确率。",
                "interpretation": "线性 scalar readout 能否直接当作 count 使用。",
            },
            {
                "metric": "linear_answer_acc",
                "definition": "同一 hidden 上训练 one-vs-all linear ridge classifier 预测 `NUM_k`。",
                "interpretation": "如果 learned readout 高但 model answer 低，瓶颈在模型自己的输出头/readout。",
            },
            {
                "metric": "logit_lens_acc",
                "definition": "把中间层 query hidden 直接过 final LN 和 output head，限制在 `NUM_k` 上取 argmax。",
                "interpretation": "当前层 hidden 是否已经处在输出头能读的坐标系里。",
            },
            {
                "metric": "unembedding_adjacent_cosine",
                "definition": "ridge count direction 与平均 `W[NUM_{k+1}] - W[NUM_k]` 的 cosine。",
                "interpretation": "count feature 是否沿着答案 token 序列的 unembedding 方向排列。",
            },
            {
                "metric": "mean_answer_margin",
                "definition": "`logit(NUM_true) - max_wrong_NUM_logit`。",
                "interpretation": "正确答案相对其他数字 token 的安全边际。",
            },
        ]
    )
    return markdown_table(rows, max_rows=30)


def main_result(summary: pd.DataFrame, checkpoint: pd.DataFrame, probe: pd.DataFrame) -> str:
    row = summary.iloc[0]
    best_count = (
        probe.sort_values(["step", "count_r2"], ascending=[True, False])
        .groupby("step", as_index=False)
        .first()[["step", "layer", "count_r2", "ridge_round_acc"]]
        .rename(columns={"layer": "best_count_layer", "ridge_round_acc": "best_count_ridge_round_acc"})
    )
    best_logit = (
        probe.sort_values(["step", "logit_lens_acc"], ascending=[True, False])
        .groupby("step", as_index=False)
        .first()[["step", "layer", "logit_lens_acc"]]
        .rename(columns={"layer": "best_logit_lens_layer"})
    )
    best_linear = (
        probe.sort_values(["step", "linear_answer_acc"], ascending=[True, False])
        .groupby("step", as_index=False)
        .first()[["step", "layer", "linear_answer_acc"]]
        .rename(columns={"layer": "best_linear_layer"})
    )
    merged = (
        checkpoint[["step", "answer_acc", "num_restricted_acc", "mae", "mean_answer_margin"]]
        .merge(best_count, on="step")
        .merge(best_logit, on="step")
        .merge(best_linear, on="step")
    )
    text = (
        f"最终 checkpoint 的 full-vocab answer accuracy 为 `{row['answer_acc']:.3f}`，"
        f"NUM-restricted accuracy 为 `{row['num_restricted_acc']:.3f}`。"
        f"best-count layer `{int(row['best_count_layer'])}` 的 count R2 为 `{row['best_count_r2']:.3f}`；"
        f"best logit-lens layer `{int(row['best_logit_lens_layer'])}` 的 logit-lens accuracy 为 `{row['best_logit_lens_acc']:.3f}`。"
        f"final layer 的 count R2 是 `{row['final_layer_count_r2']:.3f}`，logit-lens accuracy 是 `{row['final_layer_logit_lens_acc']:.3f}`。\n\n"
        "这张表把行为和 best-layer hidden/readout 指标放在同一时间轴上：\n\n"
    )
    return text + markdown_table(round_frame(merged), max_rows=30)


def clean_checkpoint_table(checkpoint: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "step",
        "train_loss",
        "loss",
        "answer_acc",
        "num_restricted_acc",
        "off_by_one_rate",
        "mae",
        "bias_pred_minus_true",
        "mean_answer_margin",
        "non_num_top1_rate",
        "non_num_mass",
    ]
    return round_frame(checkpoint[columns])


def final_layerwise_section(probe: pd.DataFrame) -> str:
    final = probe.loc[probe["step"] == probe["step"].max()].copy()
    final = final[
        [
            "layer",
            "count_r2",
            "count_mae",
            "ridge_round_acc",
            "linear_answer_acc",
            "logit_lens_acc",
            "logit_lens_margin",
            "unembedding_adjacent_cosine",
        ]
    ].sort_values("layer")
    return (
        "这一节看最终 checkpoint 每一层 query-position hidden 的状态。"
        "`count_r2 / ridge_round_acc / linear_answer_acc` 是新训练的 probe/readout；"
        "`logit_lens_acc` 和 `unembedding_adjacent_cosine` 则直接检查模型自己的 output head 是否能读。\n\n"
        + markdown_table(round_frame(final), max_rows=30)
    )


def steering_section(steering: pd.DataFrame) -> str:
    if steering.empty:
        return "没有找到 `steering.csv`。"
    layer = int(steering["layer"].iloc[0])
    baseline = steering.loc[np.isclose(steering["beta_axis_std"], 0.0)].iloc[0]
    pos = steering.loc[steering["beta_axis_std"] > 0].sort_values("beta_axis_std").tail(1).iloc[0]
    neg = steering.loc[steering["beta_axis_std"] < 0].sort_values("beta_axis_std").head(1).iloc[0]
    return (
        f"这里在 final checkpoint 的 best probe layer `{layer}` 上做 query-position steering："
        "`h' = h + beta * std(axis) * normalize(w_count)`，然后直接过 output head 看 `NUM_k` logits。\n\n"
        f"- baseline beta=0 的 answer accuracy 是 `{baseline['answer_acc']:.3f}`，"
        f"mean answer margin 是 `{baseline['mean_answer_margin']:.3f}`。\n"
        f"- 最大正向 beta 的平均预测 count shift 是 `{pos['mean_pred_count_delta_vs_baseline']:.3f}`。\n"
        f"- 最大负向 beta 的平均预测 count shift 是 `{neg['mean_pred_count_delta_vs_baseline']:.3f}`。\n\n"
        "注意：这是 final query hidden 的 logit-level steering，不是完整 forward 中替换早层 activation 后继续跑后续层。"
        "它能检验 output head 是否沿 count direction 排列，但还不能单独证明早层 count direction 在完整 computation 中因果控制答案。"
    )


def clean_steering_table(steering: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "layer",
        "beta_axis_std",
        "answer_acc",
        "mean_pred_count",
        "mean_pred_count_delta_vs_baseline",
        "frac_pred_plus_one_vs_baseline",
        "frac_pred_minus_one_vs_baseline",
        "mean_answer_margin",
        "non_num_mass",
    ]
    return round_frame(steering[columns])


def interpretation(summary: pd.DataFrame, checkpoint: pd.DataFrame, probe: pd.DataFrame, steering: pd.DataFrame) -> str:
    row = summary.iloc[0]
    final = probe.loc[probe["step"] == probe["step"].max()].copy()
    best_count = final.sort_values("count_r2", ascending=False).iloc[0]
    best_logit = final.sort_values("logit_lens_acc", ascending=False).iloc[0]
    step0 = probe.loc[probe["step"] == probe["step"].min()].sort_values("count_r2", ascending=False).iloc[0]
    step0_behavior = checkpoint.loc[checkpoint["step"] == checkpoint["step"].min()].iloc[0]
    gap = float(best_logit["linear_answer_acc"] - row["answer_acc"])
    if row["answer_acc"] >= 0.8 and best_count["count_r2"] >= 0.8:
        regime = "当前 smoke setting 已经同时学会 hidden count 和 direct answer readout。"
    elif best_count["count_r2"] >= 0.8 and gap > 0.2:
        regime = "当前 setting 出现 hidden readable 但模型输出头/readout 没完全接上的 gap。"
    else:
        regime = "当前 setting 还没有稳定形成足够强的 hidden count 或 direct answer readout。"

    steering_text = ""
    if not steering.empty:
        positive = steering.loc[steering["beta_axis_std"] > 0, "mean_pred_count_delta_vs_baseline"].max()
        negative = steering.loc[steering["beta_axis_std"] < 0, "mean_pred_count_delta_vs_baseline"].min()
        steering_text = (
            f"\n\nsteering 方向检查显示：正向最大平均 count shift 为 `{positive:.3f}`，"
            f"负向最大平均 count shift 为 `{negative:.3f}`。"
            "如果这两个数方向稳定且 non-NUM mass 没明显上升，说明 output head 至少在 query hidden 层面使用了 count axis。"
        )

    return (
        f"{regime}\n\n"
        f"更具体地说，最终 best-count layer 是 `{int(best_count['layer'])}`，count R2=`{best_count['count_r2']:.3f}`，"
        f"ridge round acc=`{best_count['ridge_round_acc']:.3f}`；best-output/logit-lens layer 是 `{int(best_logit['layer'])}`，"
        f"logit-lens acc=`{best_logit['logit_lens_acc']:.3f}`，linear answer acc=`{best_logit['linear_answer_acc']:.3f}`。"
        f"模型真实 answer acc=`{row['answer_acc']:.3f}`，best-output learned readout 与模型真实输出的 gap 是 `{gap:.3f}`。"
        f"\n\n一个关键现象是 step 0 时 best count R2 已经有 `{step0['count_r2']:.3f}`，"
        f"但 answer acc 只有 `{step0_behavior['answer_acc']:.3f}`。这说明在这个 NeedleCount smoke task 里，"
        "随机初始化的 query hidden 已经能线性反映 token occurrence count；训练真正完成的是把这个可读 count "
        "搬到 output head 能稳定读出的 final-layer answer-token 坐标系。"
        f"{steering_text}"
    )


def next_steps() -> str:
    return (
        "1. 扩展正式 grid：`seq_len=[128,512,2000]`、`max_count=[10,30]`、多 seed。\n"
        "2. 加高 distractor ratio / sparse occurrence setting，测试是否出现类似 Task A 的 hidden-readable 但 output-failed regime。\n"
        "3. 把 direct count task 接入正式 sampler/config/checkpoint pipeline，和 A/B 的 Transformer baseline 保持完全同构。\n"
        "4. 做真正 layer-wise activation patch：在早层替换或沿 count direction steering 后继续跑后续层，而不是只做 final query logit steering。\n"
        "5. 对比 DyckCounter direct count 与 NeedleCount direct count，区分 formal stack counter 和 occurrence retrieval/counting。"
    )


def markdown_table(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame.empty:
        return "_empty table_"
    shown = frame.head(max_rows).copy()
    shown = shown.fillna("NA")
    headers = list(shown.columns)
    lines = [
        "| " + " | ".join(str(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in shown.iterrows():
        lines.append("| " + " | ".join(format_value(row[col]) for col in headers) + " |")
    if len(frame) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(frame)} rows._")
    return "\n".join(lines)


def round_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rounded = frame.copy()
    for column in rounded.columns:
        if pd.api.types.is_float_dtype(rounded[column]):
            rounded[column] = rounded[column].map(lambda value: np.nan if pd.isna(value) else round(float(value), 4))
    return rounded


def format_value(value: object) -> str:
    if isinstance(value, float):
        if np.isnan(value):
            return "NA"
        return f"{value:.4g}"
    return str(value)


def rel_from_notebook(output: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), start=output.parent.resolve())


if __name__ == "__main__":
    main()
