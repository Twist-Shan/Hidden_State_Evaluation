from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "Dyck_Syn_to_Rea" / "Task_A_Length_Noise.ipynb"
MARKER = "TASK_A_REWRITTEN_ANALYSIS"


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") == "markdown" and MARKER in source:
            cell["source"] = intro_text().splitlines(keepends=True)
            break
    else:
        nb["cells"].insert(0, {"cell_type": "markdown", "metadata": {}, "source": intro_text().splitlines(keepends=True)})
    NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"updated {NOTEBOOK}")


def intro_text() -> str:
    return (
        f"<!-- {MARKER} -->\n"
        "# Dyck Syn-to-Rea: Task A Length and Noise\n\n"
        "这个 notebook 的问题是：Transformer 在带 noise 的 Dyck-1 synthetic sequence 上，"
        "到底是学到了 Dyck 计数机制，还是只学到了一些局部 readout / shortcut？\n\n"
        "阅读顺序如下。\n\n"
        "1. **实验设置与主结果。** 先看 6 个 Transformer settings 的 behavior、oracle forced/free split、"
        "以及 hidden-state count probe。这里的核心结论是：普通 settings 接近 stochastic generator 的 oracle ceiling；"
        "`tiny_extreme_long/b20` 是真正失败的稀疏监督 regime。\n\n"
        "2. **固定 seq_len=2000 的 sparsity sweep。** 再只改变 bracket token 数量，定位 b20 到 b100 之间的转变。"
        "这一段说明：forced open/close 规则其实较早能在 bracket 子空间中读出，但 full-vocab 输出需要足够 bracket mass。\n\n"
        "3. **基础 probes 和 ablations。** 然后检查 height/left/right 是否线性可读、是否跨 setting 稳定、"
        "以及 height direction 是否直接接到 output head。这里的核心区分是：可线性读出不等于被模型用于输出。\n\n"
        "4. **CountScope-style online patching。** 这一段做 source/target/patched forward pass。"
        "`local_interchange` 测当前位置 readout 是否能被 hidden state 改写；"
        "`future_continued` 测较早 patch 是否能影响后续 forced decision。"
        "结果支持 late hidden 对局部 open/close readout 有因果作用，但不支持稳定的 continued counter-state transfer。\n\n"
        "5. **Axis/span patching 与 sparse failure diagnostics。** 最后把 readout 方向、Dyck-target CE/acc、"
        "forced/free split、bracket-only 指标和 bracket mass 放在一起看。"
        "这一段是目前最重要的机制解释：模型确实表示 count/left/right；但 next-bracket 输出主要接在与 output head 对齐的 forced-decision direction 上。"
        "低 density 失败主要来自 bracket token 在 full vocab 中输给 noise，而不是单纯 open/close 子空间失效。\n\n"
        "因此，看这个 notebook 时不要只盯 overall accuracy。更可靠的读法是同时看："
        "`forced/free split`、`Dyck-target full-vocab CE/acc`、`bracket-only CE/acc`、`bracket mass`，"
        "以及 patching/ablation 是否显示某个方向真的被 output path 使用。"
    )


if __name__ == "__main__":
    main()
