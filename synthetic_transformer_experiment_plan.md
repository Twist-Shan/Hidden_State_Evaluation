# Synthetic Transformer 实验计划：从 counting feature 到输出结果

本文档根据 `Chatting/research-plan-and-findings.html` 中 `Research plan v2 -> To do list -> Synthetic experiment` 的方向整理。主体模型固定为 Transformer，目标不是简单扩展 Dyck baseline，而是用可控 synthetic task 去贴近 realistic setting 中的关键猜想：

1. 长上下文和噪声会怎样破坏 counting behavior？
2. Transformer 什么时候形成可线性解码的 counting feature？
3. internal counting feature 和准确输出之间是否存在时间差、层差和机制差？
4. 模型如何把 hidden-state 中的 count 转换成最终展示结果，例如 next-token、JSON count token 或 reasoning token？
5. synthetic task 能否区分 retrieval 机制和 internal counter 机制？

## 0. 总体设计原则

### 0.1 论文问题

这组实验要服务于一个更大的 paper story：

> Counting information can be linearly readable in Transformer hidden states before, and sometimes without, being reliably verbalized as the correct output. Synthetic tasks let us control context length, noise, structure, and supervision format, so we can locate when the feature emerges and when it becomes usable for prediction.

对应到 NIAH 的 realistic 猜想：

- NIAH 里已经看到 `probe R2` 很高，但 direct answer 不稳定。
- thinking/CoT 让模型通过显式计数和有序检索变得稳定。
- activation patching 更像是 retrieval of occurrences，而不是纯粹压缩后的 internal counter。
- synthetic task 要补足 NIAH 难以回答的问题：训练过程中 feature 何时出现？feature 到输出的 readout 何时学会？噪声和长度如何影响二者的 gap？

### 0.2 默认 Transformer 设置

优先沿用 Hidden State Evaluation 当前 pipeline 的小 Transformer，保证多 seed、多 checkpoint、多 probe 可跑。

| 项目 | 默认值 | 说明 |
|---|---:|---|
| architecture | decoder-only Transformer | causal next-token prediction |
| layers | `3` | 与现有 baseline 对齐 |
| `d_model` / `emb_dim` | `128` | 现有计划默认值 |
| heads | `4` | 每头 32 dim |
| `d_ff` | `512` | FFN expansion 4x |
| positional encoding | 当前实现默认，必要时记录 absolute/sinusoidal | 长度外推时必须固定说明 |
| dropout | `0.0` for clean mechanistic runs, `0.1` for robustness check | 训练动态主实验建议先用 `0.0` |
| optimizer | AdamW |
| learning rate | `3e-4` |
| batch size | `128` for length <= 400, `64` for 1000, `32` for 2000 | 以显存为准 |
| seeds | `[0, 1, 2]` | 核心实验必须三 seed |
| checkpoints | `[0, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 15000, 30000]` | emergence 分析必须保存 |

可选 scale-up control：

| 项目 | 值 |
|---|---:|
| layers | `6` |
| `d_model` | `256` |
| heads | `8` |
| `d_ff` | `1024` |
| seeds | `[0]` first, then `[0,1,2]` if needed |

scale-up 不是主线，只用于确认小模型发现不是 architecture-capacity artifact。

### 0.3 统一记录的指标

每个 run 都需要保存以下信息，避免后面只能看 accuracy：

| 类别 | 指标 |
|---|---|
| behavior | train/test loss, perplexity, next-token accuracy, bracket-only accuracy, exact count accuracy, off-by-one rate, undercount/overcount bias |
| probe | layerwise `R2`, MAE, rounded accuracy, class accuracy, probe direction norm, cross-seed direction cosine |
| gap | `probe_score - behavior_score`, `probe_emergence_step`, `behavior_emergence_step`, `verbalization_lag` |
| readout | count direction vs unembedding cosine, logit-lens count accuracy, answer-token margin, causal steering effect |
| compression | task-relevant probe score, noise-token probe score, position probe score, irrelevant-token leakage |
| attention | attention mass to relevant tokens, ordered retrieval score, last-token vs occurrence-token attention distribution |
| intervention | activation patching effect on predicted count, restore effect, count-direction steering effect |

## 1. 核心实验 A：长度和噪声如何影响 counting

### 1.1 目的

对应老师 To-Do 的第一条：

> Model accuracy as context becomes longer and noisier.

我们要分开测两件事：

1. behavior 是否因为长度和噪声下降？
2. hidden state 中的 counting feature 是否也一起下降，还是 behavior 先掉、feature 仍然可读？

如果后者成立，就支持 realistic NIAH 中的主要现象：模型内部有 count，但不会稳定展示出来。

### 1.2 任务 A1：`DyckCounter-Scalar`

这是现有 Dyck 的系统化扩展。序列由 Dyck-1 括号和 noise token 组成，模型做 causal next-token prediction。

需要记录每个 prefix 的标签：

- `left`: 已见 `(` 数量。
- `right`: 已见 `)` 数量。
- `height = left - right`。
- `dyck_index`: 当前是第几个括号 token，不算 noise。
- `is_noise_prefix_ratio`: prefix 中 noise 比例。
- `legal_next_class`: 下一步合法 token 类型。

参数网格：

| factor | values |
|---|---|
| bracket pairs | `[24, 100, 200]` |
| bracket tokens | `[48, 200, 400]` |
| context length | `[48, 120, 400, 1000, 2000]` |
| repeat/generation prob | `[1.0, 0.5, 0.25, 0.1]` |
| noise vocab size | `[4, 16, 64]` |
| train steps | `15000` main, `30000` for hard cases |
| seeds | `[0, 1, 2]` |

建议第一批不要全网格，先跑：

| run group | pairs | context length | generation prob | noise vocab | seeds |
|---|---:|---:|---:|---:|---|
| clean-short | 24 | 48 | 1.0 | 4 | 0 |
| noisy-short | 24 | 120 | 0.5 | 16 | 0 |
| sparse-medium | 100 | 400 | 0.25 | 16 | 0 |
| sparse-long | 200 | 1000 | 0.25 | 16 | 0 |
| extreme-long | 200 | 2000 | 0.1 | 64 | 0 |

### 1.3 分析内容

每个 checkpoint、layer、position 都训练 ridge/logistic probe：

| target | probe | metric |
|---|---|---|
| `left` | Ridge | `R2`, MAE |
| `right` | Ridge | `R2`, MAE |
| `height` | Ridge | `R2`, MAE, rounded accuracy |
| `height_class` | Logistic | accuracy |
| `left_right_class` | Logistic | accuracy |
| `legal_next_class` | Logistic | accuracy |
| noise prefix identity | Logistic or probe subset | compression leakage |

关键图：

1. `context length x noise level -> test loss / bracket-only accuracy`。
2. `context length x noise level -> best layer height R2`。
3. behavior vs probe scatter：每个点是一个 run/checkpoint。
4. `gap = height_probe_R2 - normalized_behavior_accuracy` 随长度和噪声变化。
5. layerwise heatmap：不同 layer 在不同 count range 的 decoding quality。

### 1.4 预期判别

支持 NIAH-style gap 的结果：

- 长度/噪声增加时，next-token 或 final count accuracy 明显下降。
- 同时 `height` 或 `left/right` probe 仍然保持较高 `R2`。
- gap 在高噪声、长上下文、late checkpoint 中最大。

不支持该猜想的结果：

- behavior 下降时 probe 也同步下降，说明主要瓶颈是 representation 没形成，而不是 readout/verbalization。

## 2. 核心实验 B：counting feature 什么时候出现

### 2.1 目的

对应老师 To-Do 的第二条：

> When does the model form internal counting feature, as revealed by ridge regression? Training dynamics analysis.

这里要把“出现”定义清楚，不能只说“probe 变高”。

### 2.2 emergence 定义

对每个 run、layer、target 定义：

| 名称 | 定义 |
|---|---|
| `probe_emergence_step` | 第一个满足 `test R2 >= 0.8` 且 `rounded_acc >= 0.5` 的 checkpoint |
| `stable_probe_step` | 从该 checkpoint 起到训练结束，`R2` 大部分保持在阈值以上 |
| `behavior_emergence_step` | bracket-only accuracy 或 exact count accuracy 第一次超过 `0.8` 的 checkpoint |
| `verbalization_lag` | `behavior_emergence_step - probe_emergence_step` |
| `early_feature_layer` | 最早出现高 probe score 的 layer |
| `best_feature_layer` | 最终 checkpoint probe score 最高的 layer |

阈值可以做 sensitivity check：

- `R2 >= 0.7 / 0.8 / 0.9`。
- `exact count accuracy >= 0.7 / 0.8 / 0.9`。

### 2.3 训练动态任务

选择 4 个代表性 task，而不是所有网格都保存大量 checkpoint：

| task | params | 目的 |
|---|---|---|
| `DyckCounter-clean` | pairs 24, context 48, prob 1.0 | 最简单计数，feature 应最早出现 |
| `DyckCounter-noisy` | pairs 24, context 120, prob 0.5 | 现有 50% noise baseline |
| `DyckCounter-long` | pairs 200, context 1000, prob 0.25 | 长上下文压力 |
| `NeedleCount-synthetic` | max count 30, context 2000, distractor 0.9 | 对齐 NIAH |

每个任务保存 checkpoint：

```text
0, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 15000, 30000
```

每个 checkpoint 抽取：

- embedding output。
- layer 1, 2, 3 residual stream。
- attention output and MLP output if current code方便。
- final token、all bracket positions、fixed prefix fractions `[0.25, 0.5, 0.75, 1.0]`。

### 2.4 分析问题

要回答的具体问题：

1. `height` feature 是否先于 next-token behavior 出现？
2. `left/right` 是否比 `height` 更早出现，还是模型直接学 `height`？
3. 低层是否先表示小 count，高层是否后续表示大 count？
4. 长上下文和 noise 是否推迟 `probe_emergence_step`，还是主要推迟 `behavior_emergence_step`？
5. checkpoint 之间 probe direction 是否稳定，还是早期 feature 和后期 feature 是不同方向？

关键图：

- `step x layer -> height R2` heatmap。
- `step x layer -> exact count accuracy` heatmap。
- `step -> probe R2` 和 `step -> behavior accuracy` 双曲线。
- `step -> cosine(w_height_step, w_height_final)`。
- count range 分段图：small count `[0,5]`、medium `[6,15]`、large `[16,30]`。

## 3. 核心实验 C：从 counting feature 到输出展示的转换

### 3.1 目的

对应老师 To-Do 的第三条：

> Does the model first learn internal counting feature, then later translates the internal feature to improve prediction?

这里的核心是 readout/verbalization。需要把 hidden count 和 output logits 接上，而不只是 probe hidden state。

### 3.2 任务 C1：Direct final count

在 Dyck 或 NeedleCount 序列之后追加 query token，模型需要输出一个 count token。

格式建议：

```text
<seq> <QUERY_COUNT> <ANS_0/ANS_1/.../ANS_30>
```

或 JSON-like：

```text
<seq> <QUERY_JSON> { count : <NUM_k> }
```

第一版建议用 single-token count label，避免 tokenizer 和多位数字干扰。

参数：

| factor | values |
|---|---|
| source task | `DyckCounter`, `NeedleCount-synthetic` |
| max count | `[10, 30]` |
| context length | `[128, 512, 2000]` |
| noise/distractor ratio | `[0.0, 0.5, 0.9]` |
| output vocab | `NUM_0 ... NUM_30` |
| train steps | `30000` |
| seeds | `[0,1,2]` |

### 3.3 readout 对齐分析

对每个 checkpoint/layer/position 计算：

| 分析 | 具体做法 | 解释 |
|---|---|---|
| count probe | `w_count` from ridge | hidden count direction |
| unembedding alignment | cosine between `w_count` and `W_U[NUM_k+1] - W_U[NUM_k]` or fitted answer direction | count feature 是否直接对齐输出空间 |
| logit lens | 用中间层 hidden 直接过 final LN/unembedding 看 count token logits | count 信息何时已经可被输出头读取 |
| learned readout | 训练 linear map from hidden to answer logits | hidden 中有信息但原输出头不用时，learned readout 会高于 model answer |
| answer margin | `logit(NUM_true) - max_wrong_logit` | 是否只是差一点输出正确 |

关键判别：

- 如果 `probe R2` 高但 unembedding alignment 低，说明 feature 可读但没有和输出 token 对齐。
- 如果 learned readout 高但原模型 answer 低，说明 bottleneck 在 model readout/verbalization。
- 如果 logit lens 在 late layer 才变好，说明 count feature 到 answer token 的转换发生在后层。

### 3.4 causal steering

用 ridge direction 做干预，但要严格避免“看起来能动一点”的弱结论。

干预位置：

- occurrence positions。
- final query position。
- answer-before position，也就是要预测 `NUM_k` 的位置。

干预方式：

```text
h' = h + beta * normalize(w_count)
```

beta grid：

```text
[-5, -3, -1, -0.5, 0.5, 1, 3, 5]
```

记录：

- predicted count 是否 `+1` 或 `-1`。
- true answer logit margin 是否增加。
- non-count token loss 是否被破坏。
- held-out set 上是否有效，而不是只在构造 probe 的 examples 上有效。

结论标准：

- 强证据：steering 在 held-out examples 上系统性移动 count logits，而且不显著破坏其它 token。
- 弱证据：只在少数 layer/beta 上改变个别失败样本，不能作为主要结论。

## 4. 核心实验 D：Synthetic NIAH bridge

### 4.1 目的

把 synthetic task 贴近 realistic NIAH。Dyck 是干净的 formal language，NIAH 是长上下文中的 repeated evidence counting。Synthetic NIAH bridge 用可控 token 复现 NIAH 的变量：

- needles 数量。
- context length。
- distractor density。
- needle span length。
- confusable distractors。
- direct answer vs reasoning answer。

### 4.2 任务 D1：`NeedleCount-synthetic`

序列由 distractor token 和 needle span 组成。模型最后回答 needle 数量。

例子：

```text
N_1 D_4 D_2 N_1 D_7 N_1 <QUERY_COUNT> NUM_3
```

更接近 NIAH 的 span 版本：

```text
CITY_A SCORE_A D D CITY_B SCORE_B D CITY_C SCORE_C <QUERY_COUNT> NUM_3
```

labels：

- total needle count。
- occurrence index at each needle span：第几个 needle。
- prefix count at every token。
- needle positions。
- distractor ratio。
- last needle distance。

参数网格：

| factor | values |
|---|---|
| max count | `[5, 10, 30]` |
| context length | `[128, 512, 2000, 10000]` |
| distractor ratio | `[0.0, 0.5, 0.9, 0.98]` |
| needle span length | `[1, 4, 12]` |
| distractor vocab size | `[16, 128, 1024]` |
| confusable distractor rate | `[0.0, 0.2]` |
| answer format | `NUM_k`, JSON-like count |
| train steps | `30000` |
| seeds | `[0,1,2]` for <= 2000, `[0]` for 10000 first |

### 4.3 对齐 NIAH 的 probe

复用 NIAH 的两类 probe：

| NIAH probe | Synthetic 对应 |
|---|---|
| sequence-level final-token probe | query position hidden -> total count |
| mean needle-span probe | average hidden over all needle spans -> total count |
| occurrence-level probe | each needle span hidden -> occurrence index |
| last-token probe orthogonality | compare final token direction with occurrence direction |

需要额外做：

- probe direction cosine between occurrence-level and final-token probes。
- large count undercount analysis，尤其 max count 30。
- position sensitivity：needle near beginning vs middle vs end。

### 4.4 预期结果

如果 synthetic bridge 成功，它应该复现 NIAH 的几个现象：

- max count 10 时 `R2` 很高。
- max count 30 时仍可读，但大 count 有 undercount bias。
- direct answer accuracy 比 probe score 更脆弱。
- final-token direction 和 occurrence-level direction 可能正交或弱相关。
- high distractor ratio 下 behavior drop 大于 probe drop。

## 5. 核心实验 E：CoT / reasoning token 如何帮助展示 count

### 5.1 目的

对应老师 To-Do 的第四条：

> Map the sequence to clean reasoning tokens; perhaps using explicit counter (`1`, `2`, `3`). Train on original sequence appended with the reasoning tokens.

我们要判断：CoT 是让模型更好地形成 internal counter，还是让模型更好地把已有 count 展示出来？

### 5.2 三种训练格式

对同一个 source sequence，训练三种 output format。

| format | example | 问题 |
|---|---|---|
| direct | `<seq> <QUERY_COUNT> NUM_7` | hidden count 能否直接转成答案 |
| appended counter trace | `<seq> <REASON> NUM_1 NUM_2 ... NUM_7 <FINAL> NUM_7` | 显式 counter 是否降低 verbalization gap |
| occurrence-tag trace | `<seq> <REASON> POS_i NUM_1 POS_j NUM_2 ... <FINAL> NUM_7` | 是否诱导有序 retrieval |

第一版可以不用真实 `POS_i`，只生成：

```text
<REASON> COUNT_1 COUNT_2 ... COUNT_k <FINAL> NUM_k
```

第二版再加入 occurrence marker：

```text
<REASON> NEEDLE COUNT_1 NEEDLE COUNT_2 ... <FINAL> NUM_k
```

### 5.3 参数

| factor | values |
|---|---|
| source task | `NeedleCount-synthetic`, `DyckCounter-Scalar` |
| max count | `[10, 30]` |
| context length | `[512, 2000]` |
| distractor/noise ratio | `[0.5, 0.9]` |
| output format | direct, counter trace, occurrence trace |
| train steps | `30000` |
| checkpoints | full emergence list |
| seeds | `[0,1,2]` |

### 5.4 分析内容

比较 direct 和 CoT：

| 分析 | direct | CoT |
|---|---|---|
| exact final count accuracy | baseline | 是否显著提高 |
| probe emergence | 是否已早出现 | 是否更早或更强 |
| verbalization lag | 可能大 | 是否缩小 |
| final-token readout alignment | 可能弱 | 是否增强 |
| attention pattern | diffuse retrieval | 是否变成 ordered retrieval |
| intermediate count token accuracy | 无 | `COUNT_i` 是否逐步正确 |

ordered retrieval score：

- 对生成 `COUNT_i` 的位置，计算 attention mass 是否集中到第 `i` 个 needle 附近。
- 计算 attention argmax 的 needle index 是否随 reasoning step 单调增加。
- 记录 `ordered_attention_acc = mean(argmax_needle_index == i)`。

关键图：

1. direct vs CoT final accuracy over count range。
2. direct vs CoT `verbalization_lag`。
3. reasoning step `i` -> attended needle index heatmap。
4. CoT 中 `COUNT_i` token 的 layerwise logit lens。

### 5.5 预期判别

如果 CoT 主要帮助 readout：

- direct 和 CoT 的 early hidden count probe 都出现得早。
- CoT 的 final answer accuracy 更高，readout alignment 更强。
- `verbalization_lag` 在 CoT 中明显缩小。

如果 CoT 主要帮助 representation：

- CoT run 的 probe emergence 也明显更早、更强。
- direct run 在相同 step 下 hidden count 本身较弱。

如果 CoT 是 retrieval iterator：

- reasoning positions 的 attention 会按 count index 有序扫过 needles。
- occurrence-level hidden state 更像 single occurrence index，而 final answer 依赖 reasoning trace 聚合。

## 6. 需要新增或改造的代码模块

### 6.1 Configs

建议新增：

```text
configs/dyck_counter_length_noise_transformer.yaml
configs/dyck_counter_emergence_transformer.yaml
configs/needle_count_synthetic_direct.yaml
configs/needle_count_synthetic_cot.yaml
configs/shuffle_dyck_length_noise_transformer.yaml
configs/dyck_k_stack_sweep_transformer.yaml
```

每个 config 明确写：

- `model: transformer` 或只跑 transformer。
- `checkpoint_steps`。
- `extract_layers: all`。
- `extract_positions: all / final / occurrence / fixed_prefix_fraction`。
- `answer_format: next_token / direct_count / counter_trace`。

### 6.2 Task modules

现有：

- `src/hse/tasks/dyck`
- `src/hse/tasks/shuffle_dyck`
- `src/hse/tasks/dyck_k`

需要新增或补齐：

```text
src/hse/tasks/needle_count/
src/hse/tasks/dyck_counter_answer/
src/hse/tasks/reasoning_count/
```

`needle_count` sampler 必须返回：

- token ids。
- needle spans。
- occurrence index labels。
- prefix count labels at all positions。
- final count label。
- distractor metadata。

### 6.3 Analysis scripts

建议新增：

```text
scripts/extract_checkpoint_hidden_states.py
scripts/run_checkpoint_probes.py
scripts/analyze_emergence.py
scripts/analyze_readout_alignment.py
scripts/run_count_steering.py
scripts/run_occurrence_patching.py
scripts/analyze_ordered_attention.py
```

输出文件统一放：

```text
results/<run_name>/
  checkpoints/
  metrics.json
  hidden_states/
  probes/
    layerwise_probe.csv
    checkpoint_probe.csv
    probe_direction_cosine.csv
  readout/
    unembedding_alignment.csv
    logit_lens_count.csv
  interventions/
    steering_summary.csv
    patching_summary.csv
  attention/
    ordered_retrieval_summary.csv
```
## 7. 最近一周可执行清单

1. 先用现有 config 跑 Transformer-only baseline，确认 `dyck_no_noise`、`dyck_noise`、`shuffle_dyck`、`dyck_k_no_noise` 的 probe 表是完整的。
2. 加 checkpoint 保存和 checkpoint probe，先在 `DyckCounter-clean/noisy` 上跑 emergence。
3. 新增 `NeedleCount-synthetic` direct answer，先做 max count 10、context 512、distractor 0.9。
4. 对同一 setting 加 counter trace output，比较 direct vs CoT。
5. 写 `analyze_emergence.py` 输出 `probe_emergence_step`、`behavior_emergence_step`、`verbalization_lag`。
6. 写 `analyze_readout_alignment.py`，先做 count direction 与 `NUM_k` unembedding 的 alignment 和 logit lens。
7. 只在看到明显 gap 后，再做 steering 和 activation patching。