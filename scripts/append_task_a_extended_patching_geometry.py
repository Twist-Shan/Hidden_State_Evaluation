from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "Dyck_Syn_to_Rea" / "Task_A_Length_Noise.ipynb"
AXIS_RESULT_DIR = ROOT / "results" / "dyck_counter_task_a_axis_span_patching"
AXIS_FIG_DIR = ROOT / "figures" / "dyck_counter_task_a_axis_span_patching"
SPARSE_RESULT_DIR = ROOT / "results" / "dyck_counter_task_a_sparse_geometry"
SPARSE_FIG_DIR = ROOT / "figures" / "dyck_counter_task_a_sparse_geometry"
MARKER = "TASK_A_EXTENDED_PATCHING_GEOMETRY"
COUNTSCOPE_MARKER = "TASK_A_COUNTSCOPE_ONLINE_PATCHING"


def main() -> None:
    axis = pd.read_csv(AXIS_RESULT_DIR / "axis_span_patching_summary.csv")
    token_loss = pd.read_csv(SPARSE_RESULT_DIR / "sparse_token_loss.csv")
    geometry = pd.read_csv(SPARSE_RESULT_DIR / "direction_geometry.csv")
    stability = pd.read_csv(SPARSE_RESULT_DIR / "direction_stability.csv")
    ablation = pd.read_csv(SPARSE_RESULT_DIR / "direction_ablation.csv")
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    nb["cells"] = [cell for cell in nb["cells"] if cell.get("metadata", {}).get("marker") != MARKER]
    insert_after_marker(
        nb["cells"],
        COUNTSCOPE_MARKER,
        build_cells(axis, token_loss, geometry, stability, ablation),
    )
    NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"updated {NOTEBOOK}")


def insert_after_marker(cells: list[dict], marker: str, new_cells: list[dict]) -> None:
    last_index = None
    for index, cell in enumerate(cells):
        if cell.get("metadata", {}).get("marker") == marker:
            last_index = index
    if last_index is None:
        cells.extend(new_cells)
    else:
        cells[last_index + 1 : last_index + 1] = new_cells


def build_cells(
    axis: pd.DataFrame,
    token_loss: pd.DataFrame,
    geometry: pd.DataFrame,
    stability: pd.DataFrame,
    ablation: pd.DataFrame,
) -> list[dict]:
    axis_table = axis_compact_table(axis)
    metric_snapshot = sparse_metric_snapshot(token_loss)
    split_snapshot = sparse_split_snapshot(token_loss)
    alignment_table = alignment_table_final(geometry)
    ablation_table = ablation_result_table(ablation)
    cells = [
        md(
            "## Extended Patching and Sparse Failure Diagnostics\n\n"
            f"<!-- {MARKER} -->\n\n"
            "这一节补两个更精细的问题。\n\n"
            "第一，前面的 full-state online patch 能改变 bracket readout，但还不能说明是哪条方向在起作用。"
            "所以这里做 axis/span patching：只把 source-target 在某个方向或低维子空间上的投影 copy 到 target，然后继续后续 layers。\n\n"
            "第二，固定 `seq_len=2000` 改变 bracket sparsity 时，低 density 模型为什么差？"
            "这里同时看 full-vocab CE、bracket-only CE、bracket token mass、方向夹角，以及 final-hidden direction ablation。"
        ),
        md("### Axis/Span Patching: 设计\n\n" + axis_method_text()),
        md("### Axis/Span Patching: 结果\n\n" + markdown_table(axis_table, max_rows=80)),
        md(
            "![axis_span_bracket_ci]("
            + relpath(AXIS_FIG_DIR / "axis_span_bracket_ci.png")
            + ")\n\n"
            "![axis_span_full_vocab_ci]("
            + relpath(AXIS_FIG_DIR / "axis_span_full_vocab_ci.png")
            + ")\n\n"
            "![axis_span_best_layer_heatmap]("
            + relpath(AXIS_FIG_DIR / "axis_span_best_layer_heatmap.png")
            + ")"
        ),
        md("### Sparse Sweep: Accuracy/CE 指标定义\n\n" + metric_text()),
        md(
            "### Sparse Sweep: 关键结果\n\n"
            + sparse_metric_results_text(token_loss)
            + "\n\nDyck-target metrics, selected densities:\n\n"
            + markdown_table(metric_snapshot, max_rows=20)
            + "\n\nForced/free split, selected densities:\n\n"
            + markdown_table(split_snapshot, max_rows=20)
        ),
        md(
            "![sparse_loss_behavior]("
            + relpath(SPARSE_FIG_DIR / "sparse_loss_behavior.png")
            + ")"
        ),
        md("### Direction Geometry and Final-hidden Ablation\n\n" + geometry_method_text()),
        md("Final-layer key alignments, selected densities:\n\n" + markdown_table(alignment_table, max_rows=80)),
        md("Forced-split final-hidden direction ablation, selected densities:\n\n" + markdown_table(ablation_table, max_rows=80)),
        md(
            "![sparse_direction_alignment]("
            + relpath(SPARSE_FIG_DIR / "sparse_direction_alignment.png")
            + ")\n\n"
            "![sparse_direction_stability]("
            + relpath(SPARSE_FIG_DIR / "sparse_direction_stability.png")
            + ")\n\n"
            "![sparse_direction_ablation]("
            + relpath(SPARSE_FIG_DIR / "sparse_direction_ablation.png")
            + ")"
        ),
        md("### 结果解释\n\n" + interpretation(axis, token_loss, geometry, ablation)),
        code(
            "# Re-run this block if needed\n"
            "# !conda run -n hse python scripts/task_a_axis_span_patching.py --settings b20 b48 b100 --pairs-per-setting 256 --batch-size 8 --max-batches 260 --device cuda\n"
            "# !conda run -n hse python scripts/task_a_sparse_geometry_diagnostics.py --max-rows 160000\n"
            "# !conda run -n hse python scripts/append_task_a_extended_patching_geometry.py"
        ),
    ]
    for cell in cells:
        cell.setdefault("metadata", {})["marker"] = MARKER
    return cells


def axis_method_text() -> str:
    modes = pd.DataFrame(
        [
            ("full_source", "整段 source hidden state 替换 target hidden state。"),
            ("full_source_shuffle", "打乱 source-target pairing 的 full-state control。"),
            ("output_close_open_scalar", "只 copy source 在 output head 的 close-minus-open 方向上的 scalar projection。"),
            ("output_bracket_noise_scalar", "只 copy source 在 bracket-vs-noise 输出方向上的 scalar projection。"),
            ("output_output2d_span", "copy close-open 和 bracket-vs-noise 两维 output span。"),
            ("height_scalar", "只 copy ridge height probe 方向上的 projection。"),
            ("left/right/height-left-right", "copy left/right count probe 方向或它们和 height 构成的 span。"),
            ("current_bracket_scalar", "copy 当前 token 是 close vs open 的 hidden mean-difference 方向。"),
            ("forced_next_scalar", "copy forced next close vs forced next open 的 hidden mean-difference 方向。"),
        ],
        columns=["mode", "definition"],
    )
    return (
        "仍然使用 same-model online patching：target/source 都选 forced Dyck next-token position，且 source 的 forced token 与 target 相反。"
        "patch 后继续后续 Transformer layers，然后在同一 position 解码。\n\n"
        + markdown_table(modes, max_rows=20)
        + "\n\n"
        "`bracket CI` 只在 close/open 两个 logits 上 softmax；`full-vocab CI` 在完整 vocab softmax 上计算。"
        "如果某个方向 bracket CI 高但 full-vocab CI 低，说明它能控制 open/close 相对排序，但未必能让 bracket token 赢过 noise token。"
    )


def metric_text() -> str:
    return (
        "这里不再只看一个 accuracy，因为 sparse setting 里有两个不同问题会混在一起："
        "模型是否知道下一步该 open/close，以及模型是否把足够概率分给 bracket token 而不是 noise token。\n\n"
        "- **Dyck-target**：只在下一 token 是 `open` 或 `close` 的位置上评估；不包括普通 noise 位置。\n"
        "- **forced/free split**：`forced` 是 Dyck 规则唯一决定下一 bracket 的位置；`free` 是 open/close 都合法、generator 随机采样的位置。"
        "`forced` 更适合检验模型是否学会 Dyck 约束；`free` 的上限接近随机猜测。\n"
        "- **full-vocab acc / full-vocab NLL**：在完整 vocabulary 上评估正确 bracket token。"
        "它同时受 open/close 判断和 bracket-vs-noise 竞争影响。\n"
        "- **bracket-only acc / bracket-only NLL**：只在 `open` 和 `close` 两个 logits 上重新 softmax。"
        "它隔离了 open/close 方向是否正确，不考察 bracket 是否赢过 noise。\n"
        "- **bracket mass**：完整 softmax 下 `P(open) + P(close)`。"
        "它直接衡量模型有没有把概率质量放到 bracket token 上。\n"
        "- **train eval loss**：训练 pipeline 的全 token loss。"
        "因为 `seq_len=2000` 里多数位置是 noise，它对 Dyck 行为变化不敏感。"
    )


def geometry_method_text() -> str:
    return (
        "这里把两个问题分开：\n\n"
        "1. **direction geometry**：看 ridge count directions、当前 bracket 方向、forced-next 方向，与 input/output bracket vectors 的夹角。"
        "`output_close_open = W_close - W_open`，`output_bracket_noise = mean(W_close,W_open)-mean(W_noise)`。\n\n"
        "2. **final-hidden ablation**：在最终 hidden state 上移除某个方向/子空间的 centered projection，再用同一个 output head 解码。"
        "这不是完整 online activation patch；它只回答“这个方向是否被 final output readout 因果使用”。"
        "因此它能和上面的 online patching 互相校验。"
    )


def axis_result_table(axis: pd.DataFrame) -> pd.DataFrame:
    focus = axis[axis["split"].eq("ci_valid")].copy()
    modes = [
        "full_source",
        "full_source_shuffle",
        "output_close_open_scalar",
        "output_bracket_noise_scalar",
        "output_output2d_span",
        "height_scalar",
        "left_right_span",
        "height_left_right_span",
        "current_bracket_scalar",
        "forced_next_scalar",
        "mean_forced",
        "zero",
    ]
    focus = focus[focus["mode"].isin(modes)]
    best = (
        focus.sort_values(["setting", "mode", "ci_minus_self"], ascending=[True, True, False])
        .groupby(["setting", "mode"], as_index=False)
        .first()
    )
    cols = [
        "setting",
        "bracket_tokens",
        "mode",
        "layer",
        "n",
        "ci_minus_self",
        "full_vocab_ci_minus_self",
        "patched_hypothesis_acc",
        "patched_full_hypothesis_acc",
    ]
    return round_frame(best[cols].sort_values(["bracket_tokens", "mode"]))


def axis_compact_table(axis: pd.DataFrame) -> pd.DataFrame:
    table = axis_result_table(axis)
    modes = [
        "full_source",
        "full_source_shuffle",
        "output_close_open_scalar",
        "forced_next_scalar",
        "height_scalar",
    ]
    table = table[table["setting"].isin(["b20", "b48", "b100"]) & table["mode"].isin(modes)]
    return table.sort_values(["bracket_tokens", "mode"]).reset_index(drop=True)


def token_metric_table(token_loss: pd.DataFrame) -> pd.DataFrame:
    base = token_loss[token_loss["evaluation"].eq("baseline") & token_loss["split"].eq("all_dyck_targets")].copy()
    cols = [
        "setting",
        "bracket_tokens",
        "full_vocab_acc",
        "bracket_acc",
        "full_vocab_nll",
        "bracket_nll",
        "bracket_mass",
        "train_eval_loss",
    ]
    return round_frame(base[cols].sort_values("bracket_tokens"))


def sparse_metric_snapshot(token_loss: pd.DataFrame) -> pd.DataFrame:
    base = token_loss[token_loss["evaluation"].eq("baseline")].copy()
    all_dyck = base[base["split"].eq("all_dyck_targets")].copy()
    forced = base[base["split"].eq("forced")][["setting", "full_vocab_acc"]].rename(
        columns={"full_vocab_acc": "forced_full_acc"}
    )
    out = all_dyck.merge(forced, on="setting", how="left")
    out = out[out["setting"].isin(selected_sparse_settings(out))]
    out = out[
        [
            "setting",
            "bracket_tokens",
            "full_vocab_acc",
            "bracket_acc",
            "full_vocab_nll",
            "bracket_nll",
            "bracket_mass",
            "forced_full_acc",
            "train_eval_loss",
        ]
    ].sort_values("bracket_tokens")
    return round_frame(out)


def sparse_split_snapshot(token_loss: pd.DataFrame) -> pd.DataFrame:
    base = token_loss[token_loss["evaluation"].eq("baseline")].copy()
    rows = []
    for setting in selected_sparse_settings(base):
        sub = base[base["setting"].eq(setting)]
        if sub.empty:
            continue
        first = sub.iloc[0]
        row = {"setting": setting, "bracket_tokens": int(first["bracket_tokens"])}
        for split in ["forced", "free"]:
            split_row = sub[sub["split"].eq(split)]
            if split_row.empty:
                continue
            item = split_row.iloc[0]
            row[f"{split}_full_acc"] = float(item["full_vocab_acc"])
            row[f"{split}_bracket_acc"] = float(item["bracket_acc"])
            row[f"{split}_full_nll"] = float(item["full_vocab_nll"])
            row[f"{split}_bracket_nll"] = float(item["bracket_nll"])
        rows.append(row)
    return round_frame(pd.DataFrame(rows).sort_values("bracket_tokens"))


def selected_sparse_settings(frame: pd.DataFrame) -> list[str]:
    wanted = ["b20", "b40", "b48", "b64", "b100", "b400"]
    available = set(frame["setting"].dropna().astype(str))
    return [setting for setting in wanted if setting in available]


def sparse_metric_results_text(token_loss: pd.DataFrame) -> str:
    base = token_loss[token_loss["evaluation"].eq("baseline")].copy()
    all_dyck = base[base["split"].eq("all_dyck_targets")].set_index("setting")
    forced = base[base["split"].eq("forced")].set_index("setting")
    free = base[base["split"].eq("free")].set_index("setting")

    def v(frame: pd.DataFrame, setting: str, column: str) -> str:
        if setting not in frame.index:
            return "NA"
        return fmt(frame.loc[setting, column])

    return (
        "**读法先定清楚。** `full-vocab` 指标回答“模型最终是否真的输出正确 bracket token”；"
        "`bracket-only` 指标回答“如果只在 open/close 里二选一，方向是否对”；"
        "`bracket mass` 回答“模型给 open+close 的总概率够不够”。\n\n"
        "**主要结果。** b20 的 all-Dyck full-vocab acc 只有 "
        f"{v(all_dyck, 'b20', 'full_vocab_acc')}，但 bracket-only acc 是 "
        f"{v(all_dyck, 'b20', 'bracket_acc')}，bracket mass 只有 "
        f"{v(all_dyck, 'b20', 'bracket_mass')}。"
        "这说明 b20 不是完全不会区分 open/close，而是 bracket token 在完整 vocab 里概率质量太低。\n\n"
        "**forced split 更能看出 Dyck 约束是否学会。** b20 forced bracket-only acc 已经是 "
        f"{v(forced, 'b20', 'bracket_acc')}，但 forced full-vocab acc 只有 "
        f"{v(forced, 'b20', 'full_vocab_acc')}。"
        "到 b48 forced full-vocab acc 升到 "
        f"{v(forced, 'b48', 'full_vocab_acc')}，b100 到 "
        f"{v(forced, 'b100', 'full_vocab_acc')}。"
        "这说明 sparse transition 主要发生在 bracket token 能否赢过 noise，而不是 forced open/close 规则本身。\n\n"
        "**free split 不应被解读成必须高 accuracy。** free step 的 open/close 由随机采样决定，"
        "bracket-only acc 接近 0.5 是合理的。b100 free bracket-only acc 是 "
        f"{v(free, 'b100', 'bracket_acc')}，full-vocab acc 是 "
        f"{v(free, 'b100', 'full_vocab_acc')}，接近随机 ceiling。\n\n"
        "**CE 比 accuracy 更连续，但要看 Dyck-target CE，不是全 token loss。** "
        "all-Dyck full-vocab NLL 从 b20 的 "
        f"{v(all_dyck, 'b20', 'full_vocab_nll')} 降到 b100 的 "
        f"{v(all_dyck, 'b100', 'full_vocab_nll')}，再到 b400 的 "
        f"{v(all_dyck, 'b400', 'full_vocab_nll')}；"
        "但 train eval loss 在 b20/b100 几乎不变 "
        f"({v(all_dyck, 'b20', 'train_eval_loss')} vs {v(all_dyck, 'b100', 'train_eval_loss')})。"
        "所以全 token loss 被 noise positions 稀释，不能替代 Dyck-target metrics。"
    )


def forced_metric_table(token_loss: pd.DataFrame) -> pd.DataFrame:
    base = token_loss[token_loss["evaluation"].eq("baseline") & token_loss["split"].eq("forced")].copy()
    cols = [
        "setting",
        "bracket_tokens",
        "full_vocab_acc",
        "bracket_acc",
        "full_vocab_nll",
        "bracket_nll",
        "bracket_mass",
    ]
    return round_frame(base[cols].sort_values("bracket_tokens"))


def alignment_table_final(geometry: pd.DataFrame) -> pd.DataFrame:
    final = geometry.sort_values("layer").groupby(["setting", "vector_a", "vector_b"], as_index=False).last()
    wanted = [
        ("height", "output_close_open"),
        ("height", "input_close_open"),
        ("current_bracket", "input_close_open"),
        ("forced_next", "output_close_open"),
        ("forced_next", "height"),
        ("current_bracket", "output_bracket_noise"),
    ]
    mask = np.zeros(len(final), dtype=bool)
    for a, b in wanted:
        mask |= final["vector_a"].eq(a).to_numpy() & final["vector_b"].eq(b).to_numpy()
    cols = ["setting", "bracket_tokens", "layer", "vector_a", "vector_b", "cosine", "abs_cosine", "angle_deg"]
    out = final.loc[mask, cols].copy()
    out = out[out["setting"].isin(selected_sparse_settings(out))]
    return round_frame(out.sort_values(["bracket_tokens", "vector_a", "vector_b"]))


def stability_result_table(stability: pd.DataFrame) -> pd.DataFrame:
    final = stability.sort_values("layer").groupby(["setting", "direction"], as_index=False).last()
    final = final[final["direction"].isin(["height", "left", "right", "current_bracket", "forced_next"])]
    cols = ["setting", "bracket_tokens", "layer", "direction", "abs_cosine_to_reference", "abs_cosine_to_previous_density"]
    return round_frame(final[cols].sort_values(["bracket_tokens", "direction"]))


def ablation_result_table(ablation: pd.DataFrame) -> pd.DataFrame:
    focus = ablation[
        ablation["split"].eq("forced")
        & ablation["ablation"].isin(
            [
                "remove_height",
                "remove_forced_next",
                "remove_output_close_open",
                "remove_output_bracket_noise",
                "remove_random",
            ]
        )
    ].copy()
    focus = focus[focus["setting"].isin(selected_sparse_settings(focus))]
    cols = [
        "setting",
        "bracket_tokens",
        "ablation",
        "delta_full_vocab_acc_vs_baseline",
        "delta_bracket_acc_vs_baseline",
        "delta_full_vocab_nll_vs_baseline",
        "delta_bracket_nll_vs_baseline",
        "delta_bracket_mass_vs_baseline",
    ]
    return round_frame(focus[cols].sort_values(["bracket_tokens", "ablation"]))


def interpretation(axis: pd.DataFrame, token_loss: pd.DataFrame, geometry: pd.DataFrame, ablation: pd.DataFrame) -> str:
    axis_best = axis_result_table(axis)
    token = token_metric_table(token_loss).set_index("setting")
    forced = forced_metric_table(token_loss).set_index("setting")
    align = alignment_table_final(geometry)
    align_idx = align.set_index(["setting", "vector_a", "vector_b"])
    ablated = ablation_result_table(ablation)

    def table_value(frame: pd.DataFrame, setting: str, col: str) -> str:
        if setting not in frame.index:
            return "NA"
        return fmt(frame.loc[setting, col])

    def axis_value(setting: str, mode: str, col: str) -> str:
        row = axis_best[axis_best["setting"].eq(setting) & axis_best["mode"].eq(mode)]
        if row.empty:
            return "NA"
        return fmt(row[col].iloc[0])

    def align_value(setting: str, a: str, b: str) -> str:
        key = (setting, a, b)
        if key not in align_idx.index:
            return "NA"
        return fmt(align_idx.loc[key, "abs_cosine"])

    def ablation_value(setting: str, name: str, col: str) -> str:
        row = ablated[ablated["setting"].eq(setting) & ablated["ablation"].eq(name)]
        if row.empty:
            return "NA"
        return fmt(row[col].iloc[0])

    return (
        "**1. Count probe 可读，不等于模型沿 count direction 输出。**\n\n"
        "axis/span patching 直接比较“只 patch 某条方向”是否能翻转 forced open/close。"
        "结果很明确：`output_close_open` 和 `forced_next` 方向有效，`height` 方向基本无效。"
        f"b20 的 bracket CI：output_close_open={axis_value('b20', 'output_close_open_scalar', 'ci_minus_self')}，"
        f"forced_next={axis_value('b20', 'forced_next_scalar', 'ci_minus_self')}，"
        f"height={axis_value('b20', 'height_scalar', 'ci_minus_self')}；"
        f"b100 对应为 {axis_value('b100', 'output_close_open_scalar', 'ci_minus_self')}、"
        f"{axis_value('b100', 'forced_next_scalar', 'ci_minus_self')}、"
        f"{axis_value('b100', 'height_scalar', 'ci_minus_self')}。"
        "所以 hidden state 中虽然能线性读出 height/left/right，但最终 bracket decision 主要接在另一个 readout-aligned direction 上。"
        "`full_source` 和 `full_source_shuffle` 很接近，也说明 full-state patch 的强效应不能解释成逐样本 source counter state 被精确移植。\n\n"
        "**2. sparse setting 的失败主要分成两个子问题：open/close 是否对，以及 bracket 是否赢过 noise。**\n\n"
        "b20 的 all-Dyck full-vocab acc 很低："
        f"{table_value(token, 'b20', 'full_vocab_acc')}；"
        "但 bracket-only acc 是 "
        f"{table_value(token, 'b20', 'bracket_acc')}，说明 open/close 子空间不是完全失效。"
        "真正薄弱的是 bracket mass："
        f"b20={table_value(token, 'b20', 'bracket_mass')}，"
        f"b48={table_value(token, 'b48', 'bracket_mass')}，"
        f"b100={table_value(token, 'b100', 'bracket_mass')}。"
        "因此 full-vocab accuracy 低，混合了两件事：模型是否知道 open/close，以及 bracket token 是否能在完整 vocab 中胜过 noise。"
        "后者是低 density 下的主要瓶颈。\n\n"
        "**3. 全 token eval loss 不适合判断 Dyck 行为是否学会。**\n\n"
        "b20 和 b100 的 train eval loss 几乎一样："
        f"{table_value(token, 'b20', 'train_eval_loss')} vs {table_value(token, 'b100', 'train_eval_loss')}；"
        "但 forced full-vocab acc 从 "
        f"{table_value(forced, 'b20', 'full_vocab_acc')} 升到 {table_value(forced, 'b100', 'full_vocab_acc')}。"
        "原因是 seq_len=2000 中绝大多数位置是 noise，全 token loss 被 noise prediction 主导。"
        "因此这里更应该报告 Dyck-target CE/acc、forced/free split、bracket-only 指标和 bracket mass。\n\n"
        "**4. Geometry 和 ablation 支持同一个机制解释。**\n\n"
        "final layer 中，`forced_next` 与 `output_close_open` 高度对齐："
        f"abs cosine b20={align_value('b20', 'forced_next', 'output_close_open')}，"
        f"b48={align_value('b48', 'forced_next', 'output_close_open')}。"
        "但 `height` 与 `output_close_open` 基本正交："
        f"b20={align_value('b20', 'height', 'output_close_open')}。"
        "这说明输出头读到的是 forced-decision/readout axis，不是 ridge height axis。"
        "final-hidden ablation 也一致：在 forced split 上移除 `output_close_open` 显著增加 bracket NLL "
        f"(b20 +{ablation_value('b20', 'remove_output_close_open', 'delta_bracket_nll_vs_baseline')}, "
        f"b100 +{ablation_value('b100', 'remove_output_close_open', 'delta_bracket_nll_vs_baseline')})；"
        "移除 `height` 的影响接近 0 "
        f"(b20 {ablation_value('b20', 'remove_height', 'delta_bracket_nll_vs_baseline')}, "
        f"b100 {ablation_value('b100', 'remove_height', 'delta_bracket_nll_vs_baseline')})。\n\n"
        "**结论。** 当前最稳妥的说法是：模型确实表示了 count/left/right，但 next-bracket 输出不是直接沿 height ridge direction 完成的；"
        "真正接到 output head 的是与 `output_close_open` 对齐的 forced-decision direction。"
        "低 density 的主要失败来自 full-vocab bracket-vs-noise 竞争，以及这个 forced-decision readout 是否稳定形成。"
    )


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


def markdown_table(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame.empty:
        return "_empty table_"
    shown = frame.head(max_rows).copy().fillna("NA")
    headers = list(shown.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
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
            rounded[column] = rounded[column].map(lambda x: np.nan if pd.isna(x) else round(float(x), 4))
    return rounded


def format_value(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "NA"
        return f"{float(value):.4g}"
    return str(value)


def fmt(value: object) -> str:
    if isinstance(value, pd.Series):
        value = value.iloc[0]
    if pd.isna(value):
        return "NA"
    return f"{float(value):.3f}"


def relpath(path: Path) -> str:
    return os.path.relpath(path.resolve(), start=NOTEBOOK.parent.resolve())


if __name__ == "__main__":
    main()
