from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "Dyck_Syn_to_Rea" / "Task_A_Length_Noise.ipynb"
RESULT_DIR = ROOT / "results" / "dyck_counter_task_a_countscope_patching"
FIG_DIR = ROOT / "figures" / "dyck_counter_task_a_countscope_patching"
MARKER = "TASK_A_COUNTSCOPE_ONLINE_PATCHING"
EXTENDED_MARKER = "TASK_A_EXTENDED_PATCHING_GEOMETRY"


def main() -> None:
    summary = pd.read_csv(RESULT_DIR / "countscope_online_patching_summary.csv")
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    nb["cells"] = [cell for cell in nb["cells"] if cell.get("metadata", {}).get("marker") != MARKER]
    insert_before_marker(nb["cells"], EXTENDED_MARKER, build_cells(summary))
    NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"updated {NOTEBOOK}")


def insert_before_marker(cells: list[dict], marker: str, new_cells: list[dict]) -> None:
    for index, cell in enumerate(cells):
        if cell.get("metadata", {}).get("marker") == marker:
            cells[index:index] = new_cells
            return
    cells.extend(new_cells)


def build_cells(summary: pd.DataFrame) -> list[dict]:
    source_state = source_state_table(summary)
    controls = controls_table(summary)
    cells = [
        md(
            "## CountScope-style Online Activation Patching\n\n"
            f"<!-- {MARKER} -->\n\n"
            "这一节参考 reading notes 里 activation patching / CountScope 的做法，重新补一个更谨慎的 Task A patching。"
            "核心变化是：不再把不同模型的 activation 直接互换作为主证据，而是在同一个模型内构造 source/target/patched 三个上下文，"
            "在 target forward pass 的某一层在线替换 activation，然后让后续 Transformer layer 继续计算。\n\n"
            "这次只跑固定 2000 context、noise vocab 64 的三个 sparse setting：`b20`、`b48`、`b100`。"
            "每个 setting 收集 `local_interchange` 和 `future_continued` 各 256 个 patch pair。"
        ),
        md(method_text()),
        md("### 指标定义\n\n" + metric_text()),
        md("### Main Patch Results: Source-state Mode\n\n" + markdown_table(source_state, max_rows=40)),
        md(
            "![local_interchange_ci]("
            + relpath(FIG_DIR / "local_interchange_ci.png")
            + ")\n\n"
            "![future_continued_ci]("
            + relpath(FIG_DIR / "future_continued_ci.png")
            + ")"
        ),
        md("### Controls\n\n" + controls_text() + "\n\n" + markdown_table(controls, max_rows=40)),
        md(
            "![patching_controls_summary]("
            + relpath(FIG_DIR / "patching_controls_summary.png")
            + ")\n\n"
            "![patching_full_vocab_controls_summary]("
            + relpath(FIG_DIR / "patching_full_vocab_controls_summary.png")
            + ")"
        ),
        md("### 结果说明\n\n" + interpretation(summary)),
        code(
            "# Re-run this patching block if needed\n"
            "# !python scripts/task_a_countscope_online_patching.py --settings b20 b48 b100 --pairs-per-setting 256 --batch-size 8 --max-batches 240 --device cuda\n"
            "# !python scripts/append_task_a_countscope_patching.py"
        ),
    ]
    for cell in cells:
        cell.setdefault("metadata", {})["marker"] = MARKER
    return cells


def method_text() -> str:
    return (
        "### 实验设计\n\n"
        "统一符号如下：\n\n"
        "- `C` 是 source sequence，提供被 patch 的 activation。\n"
        "- `C'` 是 target sequence，提供正常 forward 的上下文和原本支持的 target bracket。\n"
        "- `C*` 是 patched target：在 target forward pass 的某一层，把指定 position 的 hidden state 替换成 source activation，然后继续跑后续 layers 和 output head。\n\n"
        "这里的 `position` 都指当前 hidden state 所在的位置；模型用这个位置的 hidden state 预测下一 token。"
        "`target token` 是原 target sequence 真实下一 bracket；`hypothesis token` 是 patch 后我们希望模型转向的 bracket。\n\n"
        "#### 1. `local_interchange`: same-position forced decision patch\n\n"
        "**构造 pair。** 在 source 和 target 中各选一个 forced Dyck next-token position。"
        "`forced` 表示当前 Dyck 状态已经唯一决定下一 bracket：要么必须 open，要么必须 close。"
        "我们只保留 source forced bracket 与 target forced bracket 相反的 pair，例如 target 必须 open、source 必须 close。\n\n"
        "**怎么 patch。** 先正常 forward target 得到 `C'`。然后在 target 的同一个 prediction position 上，"
        "把第 `l` 层 hidden state 替换成 source 在对应 forced position 的第 `l` 层 hidden state，得到 `C*`，"
        "再继续跑后续 Transformer layers 和 output head。\n\n"
        "**怎么评估。** eval position 就是被 patch 的这个 position。"
        "如果 target 原本应该预测 open、source 表示 close，那么 hypothesis token 就是 close。"
        "我们看 patch 后 close/open readout 是否从 target token 转向 hypothesis token。\n\n"
        "**这个实验回答什么。** 它回答的是：某层 hidden state 进入模型自己的后续 computation 和 output head 后，"
        "是否足以局部改变当前 forced open/close decision。"
        "所以这是一个局部 causal decoding / interchange test。\n\n"
        "**它不能单独说明什么。** 如果这个实验阳性，只能说明 activation 中存在会被 output path 使用的 bracket-decision 信息；"
        "它还不能证明 source 的完整 counter state 被 target 后续过程继续使用。\n\n"
        "#### 2. `future_continued`: earlier-state patch plus later forced decision\n\n"
        "**构造 pair。** 在 target 中先选一个较早的 Dyck bracket position 作为 patch position，"
        "再选同一 target sequence 后面一个 forced Dyck next-token position 作为 eval position。"
        "source 提供 patch position 的 activation。\n\n"
        "**怎么定义 continued hypothesis。** 这里 hypothesis 不是 source 当前下一 token。"
        "我们把 source 在 patch position 的计数状态当作新的起点，然后接上 target 从 patch position 到 eval position 之间实际发生的 Dyck 增量：\n\n"
        "`continued_left = source_left + (target_eval_left - target_patch_left)`\n\n"
        "`continued_right = source_right + (target_eval_right - target_patch_right)`\n\n"
        "`continued_dyck_seen = source_dyck_seen + (target_eval_dyck_seen - target_patch_dyck_seen)`\n\n"
        "然后用这个 continued state 判断 eval position 是否 forced open 或 forced close。"
        "如果 continued state 给出的 forced token 与 target 原本 token 不同，这个 pair 才保留。\n\n"
        "**怎么 patch。** 在 target 的较早 patch position 替换第 `l` 层 hidden state，然后让模型继续计算后续 layers。"
        "最后不是在 patch position 解码，而是在后面的 eval position 解码。\n\n"
        "**怎么评估。** 如果 patched counter state 真的被 target 后续 computation 当作新的计数起点继续使用，"
        "那么 eval position 的输出应该从 target token 向 continued hypothesis token 移动。\n\n"
        "**这个实验回答什么。** 它更接近 notes 里的 continued counting：patch 的不是当前 readout，而是较早 latent state；"
        "真正关心的是这个 latent state 是否会影响后面位置的 forced decision。\n\n"
        "**它比 `local_interchange` 更严格。** `local_interchange` 只需要 patch 改变当前 output readout；"
        "`future_continued` 要求 patch 进去的状态被后续 sequence computation 保持并使用。"
        "所以如果 `local_interchange` 阳性但 `future_continued` 弱，比较合理的解释是：模型 late hidden 能局部控制 bracket readout，"
        "但还没有强证据说明它在做可迁移的 continued counter-state computation。"
    )


def metric_text() -> str:
    return (
        "先固定三个对象：\n\n"
        "- `C'`：target sequence 的正常 forward，不做 patch。\n"
        "- `C*`：patched target，也就是把 source activation 放进 target 后继续 forward。\n"
        "- `target token`：target sequence 原本真实的下一 bracket。\n"
        "- `hypothesis token` / `hyp`：如果 source activation 起作用，我们希望模型转向的 bracket。"
        "在 `local_interchange` 里它就是 source forced bracket；在 `future_continued` 里它是 continued state 算出的 forced bracket。\n\n"
        "还要区分两种概率空间：\n\n"
        "- `P_bracket(x)`：只拿 `close` 和 `open` 两个 logits 做 softmax 后得到的概率。"
        "它只问：如果已经限定下一 token 必须是 bracket，模型更偏 close 还是 open？\n"
        "- `P_full(x)`：在完整 vocab 上做 softmax 后得到的概率。"
        "它问：模型最终 next-token 输出真的会不会选这个 bracket token；这里 noise token 也参与竞争。\n\n"
        "#### `bracket-normalized CI`\n\n"
        "CI 是 causal influence score，衡量 patch 是否把概率从 target token 推向 hypothesis token。"
        "bracket-normalized 版本用 `P_bracket`：\n\n"
        "```text\n"
        "CI_bracket = 0.5 * [\n"
        "    P_bracket(hyp, C*)    - P_bracket(hyp, C')\n"
        "  + P_bracket(target, C') - P_bracket(target, C*)\n"
        "]\n"
        "```\n\n"
        "读法：\n\n"
        "- `CI_bracket > 0`：patch 后 close/open 子空间向 hypothesis 移动。\n"
        "- `CI_bracket ≈ 0`：patch 基本没有改变 close/open 决策。\n"
        "- `CI_bracket < 0`：patch 反而让模型更偏 target 或相反方向。\n"
        "- 这个指标**忽略 noise token**，所以它只能说明 open/close 相对排序是否被改变。\n\n"
        "#### `full-vocab CI`\n\n"
        "公式一样，但把 `P_bracket` 换成 `P_full`：\n\n"
        "```text\n"
        "CI_full = 0.5 * [\n"
        "    P_full(hyp, C*)    - P_full(hyp, C')\n"
        "  + P_full(target, C') - P_full(target, C*)\n"
        "]\n"
        "```\n\n"
        "读法：\n\n"
        "- `CI_full > 0`：patch 在完整 next-token 分布里也把概率推向 hypothesis。\n"
        "- 如果 `CI_bracket` 很高但 `CI_full` 很低，说明 patch 改变了 close/open 的相对排序，"
        "但 bracket token 本身仍然没有在 full vocab 里赢过 noise token。"
        "这正是 b20 这类 sparse setting 需要单独检查的问题。\n\n"
        "#### `ci_minus_self`\n\n"
        "`target_self` 是无操作 control：把 target 自己同一位置的 activation 再 patch 回 target。"
        "理论上这不该改变输出，但实际代码路径里仍可能有极小数值误差。\n\n"
        "```text\n"
        "ci_minus_self = CI(mode) - CI(target_self)\n"
        "```\n\n"
        "所以后面主要看 `ci_minus_self`，而不是裸 CI。它表示扣掉 patching wrapper 自身误差后的净效应。\n\n"
        "#### `patched_hypothesis_acc`\n\n"
        "patch 后，只在 close/open 两个 bracket logits 里取 argmax：\n\n"
        "```text\n"
        "patched_hypothesis_acc = mean(argmax_close_open(C*) == hypothesis token)\n"
        "```\n\n"
        "读法：它回答“patch 后，如果只在 open/close 中二选一，模型是否选了 hypothesis”。"
        "它高说明 bracket 子空间被 patch 成功翻转，但不保证 full vocab 最终输出 hypothesis。\n\n"
        "#### `patched_full_hypothesis_acc`\n\n"
        "patch 后，在完整 vocab 里取 argmax：\n\n"
        "```text\n"
        "patched_full_hypothesis_acc = mean(argmax_full_vocab(C*) == hypothesis token)\n"
        "```\n\n"
        "读法：它比 `patched_hypothesis_acc` 更严格。"
        "只有当 hypothesis bracket token 真的赢过所有 noise token 和另一个 bracket token 时，才算正确。\n\n"
        "#### 结果表里另外几个字段\n\n"
        "- `patched_target_acc`：patch 后，close/open 子空间 argmax 仍等于 target token 的比例。"
        "如果 patch 有效，这个值通常应该下降。\n"
        "- `patched_full_target_acc`：patch 后，完整 vocab argmax 仍等于 target token 的比例。\n"
        "- `mean_delta_p_hypothesis`：patch 前后 hypothesis 的 `P_bracket` 平均变化，"
        "即 `P_bracket(hyp, C*) - P_bracket(hyp, C')`。越大表示 patch 越提高 hypothesis bracket 概率。\n"
        "- `mean_delta_full_p_hypothesis`：同上，但用完整 vocab 概率 `P_full`。"
        "它能看出 patch 是否真的提高了 hypothesis token 在完整 next-token 分布里的概率。\n\n"
        "简化读法：\n\n"
        "- 想看 open/close 机制有没有被局部控制：看 `ci_minus_self` 和 `patched_hypothesis_acc`。\n"
        "- 想看最终 next-token 输出有没有真的变：看 `full_vocab_ci_minus_self` 和 `patched_full_hypothesis_acc`。\n"
        "- 想区分“open/close 方向对了”还是“bracket token 输给 noise”：比较 bracket 指标和 full-vocab 指标。"
    )


def source_state_table(summary: pd.DataFrame) -> pd.DataFrame:
    focus = summary[(summary["mode"].eq("source_state")) & (summary["split"].eq("ci_valid"))].copy()
    focus = focus[
        [
            "experiment",
            "setting",
            "layer",
            "n",
            "ci_minus_self",
            "full_vocab_ci_minus_self",
            "patched_hypothesis_acc",
            "patched_full_hypothesis_acc",
            "patched_target_acc",
            "patched_full_target_acc",
            "mean_delta_p_hypothesis",
            "mean_delta_full_p_hypothesis",
        ]
    ].sort_values(["experiment", "setting", "layer"])
    return round_frame(focus)


def controls_table(summary: pd.DataFrame) -> pd.DataFrame:
    focus = summary[
        summary["split"].eq("ci_valid")
        & summary["mode"].isin(["source_state", "source_state_shuffle", "mean_forced", "zero"])
    ].copy()
    best = (
        focus.sort_values(["experiment", "setting", "mode", "ci_minus_self"], ascending=[True, True, True, False])
        .groupby(["experiment", "setting", "mode"], as_index=False)
        .first()
    )
    best = best[
        [
            "experiment",
            "setting",
            "mode",
            "layer",
            "n",
            "ci_minus_self",
            "full_vocab_ci_minus_self",
            "patched_hypothesis_acc",
            "patched_full_hypothesis_acc",
        ]
    ].sort_values(["experiment", "setting", "mode"])
    return round_frame(best)


def controls_text() -> str:
    return (
        "controls 的读法很重要：\n\n"
        "- `source_state` 是主 patch：source activation 的 forced token 与 target 相反。\n"
        "- `source_state_shuffle` 保留 source activation distribution，但打乱 source-target 对应关系。\n"
        "- `mean_forced` 用 forced positions 的均值 activation 替换，是 in-distribution-ish mean control。\n"
        "- `zero` 是 OOD control，只用于判断 patch sensitivity，不能当机制证据。\n\n"
        "如果 `source_state` 明显强于 shuffle/mean/zero，才支持逐样本 source state 被语义性移植。"
        "如果 source_state 和 shuffle 接近，则更像 activation distribution 或 late readout 被扰动，而不是 matched counter state transfer。"
    )


def interpretation(summary: pd.DataFrame) -> str:
    source = summary[(summary["mode"].eq("source_state")) & (summary["split"].eq("ci_valid"))].copy()
    local = source[source["experiment"].eq("local_interchange")]
    future = source[source["experiment"].eq("future_continued")]
    local_l2 = local[local["layer"].eq(2)].set_index("setting")
    future_l0 = future[future["layer"].eq(0)].set_index("setting")
    controls = summary[summary["split"].eq("ci_valid")].copy()

    def value(frame: pd.DataFrame, setting: str, column: str) -> str:
        if setting not in frame.index:
            return "NA"
        return f"{float(frame.loc[setting, column]):.3f}"

    def layer_values(frame: pd.DataFrame, setting: str, column: str) -> str:
        sub = frame[frame["setting"].eq(setting)].sort_values("layer")
        if sub.empty:
            return "NA"
        return "/".join(f"L{int(row.layer)}={float(getattr(row, column)):.3f}" for row in sub.itertuples(index=False))

    def best_value(experiment: str, setting: str, mode: str, column: str) -> str:
        sub = controls[
            controls["experiment"].eq(experiment)
            & controls["setting"].eq(setting)
            & controls["mode"].eq(mode)
        ].sort_values("ci_minus_self", ascending=False)
        if sub.empty:
            return "NA"
        return f"{float(sub.iloc[0][column]):.3f}"

    return (
        "**1. `local_interchange`: layer 越靠后，局部 open/close readout 越容易被 patch。**\n\n"
        "这个任务是在同一个 forced position 上 patch，然后马上解码，所以它主要测当前位置 hidden 对 output readout 的因果作用。"
        "source-state 的 bracket CI 随 layer 增强："
        f"b20 `{layer_values(local, 'b20', 'ci_minus_self')}`；"
        f"b48 `{layer_values(local, 'b48', 'ci_minus_self')}`；"
        f"b100 `{layer_values(local, 'b100', 'ci_minus_self')}`。"
        "因此主文报 layer 2，是因为 layer 2 最接近 output head、效应最大，最直接回答“late hidden 是否能控制 bracket readout”。"
        "这不是说 layer 1 没有效果；layer 1 已经有中等效应，但尚未达到 layer 2 的直接 readout 强度。\n\n"
        "layer 2 的 bracket 指标是强阳性："
        f"b20=`{value(local_l2, 'b20', 'ci_minus_self')}`, "
        f"b48=`{value(local_l2, 'b48', 'ci_minus_self')}`, "
        f"b100=`{value(local_l2, 'b100', 'ci_minus_self')}`；"
        "bracket-subspace hypothesis accuracy 也接近 1。"
        "结论：最后层 hidden 中存在 output path 能直接读出的 open/close decision 信息。\n\n"
        "**2. `local_interchange` 的 full-vocab 结果更保守，尤其 b20。**\n\n"
        "同样是 layer 2，full-vocab CI 远小于 bracket CI："
        f"b20=`{value(local_l2, 'b20', 'full_vocab_ci_minus_self')}`, "
        f"b48=`{value(local_l2, 'b48', 'full_vocab_ci_minus_self')}`, "
        f"b100=`{value(local_l2, 'b100', 'full_vocab_ci_minus_self')}`。"
        "b20 的 bracket 子空间几乎被翻转，但 full-vocab hypothesis accuracy 只有 "
        f"`{value(local_l2, 'b20', 'patched_full_hypothesis_acc')}`；b48/b100 是 "
        f"`{value(local_l2, 'b48', 'patched_full_hypothesis_acc')}`、"
        f"`{value(local_l2, 'b100', 'patched_full_hypothesis_acc')}`。"
        "解释：b20 的 open/close 排序可以被 patch 改动，但 bracket token 在完整 vocab 里仍常输给 noise token；"
        "b48/b100 的 full-vocab readout 明显更稳定。\n\n"
        "**3. `future_continued`: 没有稳定的 continued-counting transfer。**\n\n"
        "这个任务把 source state patch 到 target 较早位置，再看后面 forced position 是否转向 continued hypothesis。"
        "这里 layer 0/1 才有机会通过后续 Transformer layers 影响 later position；layer 2 patch 到较早 position 后已经没有后续 sequence mixing，"
        "所以 layer 2 接近 0 是预期的。"
        "source-state bracket CI 为："
        f"b20 `{layer_values(future, 'b20', 'ci_minus_self')}`；"
        f"b48 `{layer_values(future, 'b48', 'ci_minus_self')}`；"
        f"b100 `{layer_values(future, 'b100', 'ci_minus_self')}`。"
        "最有机会传播的 layer 0 也只有："
        f"b20=`{value(future_l0, 'b20', 'ci_minus_self')}`, "
        f"b48=`{value(future_l0, 'b48', 'ci_minus_self')}`, "
        f"b100=`{value(future_l0, 'b100', 'ci_minus_self')}`，full-vocab CI 基本接近 0。"
        "这说明 earlier patch 对后续 forced decision 的影响弱，不能支持“source counter state 被继续使用”的强说法。\n\n"
        "**4. Controls 限制了机制解释。**\n\n"
        "`local_interchange` 中，source_state 和 source_state_shuffle 很接近："
        f"b20 source=`{best_value('local_interchange', 'b20', 'source_state', 'ci_minus_self')}` vs shuffle=`{best_value('local_interchange', 'b20', 'source_state_shuffle', 'ci_minus_self')}`；"
        f"b48 source=`{best_value('local_interchange', 'b48', 'source_state', 'ci_minus_self')}` vs shuffle=`{best_value('local_interchange', 'b48', 'source_state_shuffle', 'ci_minus_self')}`；"
        f"b100 source=`{best_value('local_interchange', 'b100', 'source_state', 'ci_minus_self')}` vs shuffle=`{best_value('local_interchange', 'b100', 'source_state_shuffle', 'ci_minus_self')}`。"
        "因此 local 阳性更像“patch 到了一个 readout-aligned activation distribution”，"
        "而不是严格的 matched source-target semantic transfer。"
        "`future_continued` 中，shuffle/mean/zero controls 也经常接近或超过 source_state，因此 continued-counting 证据更弱。\n\n"
        "**结论。** 当前结果支持：late hidden 对当前位置 close/open readout 有强因果作用。"
        "当前结果不支持：source counter state 被 target 后续 computation 稳定继续使用。"
        "b20 的额外问题是 full-vocab bracket-vs-noise 竞争，而不是单纯 open/close 子空间失效。"
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
    if isinstance(value, float):
        if np.isnan(value):
            return "NA"
        return f"{value:.4g}"
    return str(value)


def relpath(path: Path) -> str:
    return os.path.relpath(path.resolve(), start=NOTEBOOK.parent.resolve())


if __name__ == "__main__":
    main()
