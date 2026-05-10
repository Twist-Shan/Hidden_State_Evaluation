# Hidden State Sufficient Statistics and Compression: Pipeline & Plan

## 1. Core Research Questions

This project aims to compare whether RNNs, LSTMs, Transformers, and maybe Mamba models learn the sufficient statistics and information required by sequence tasks, and whether they perform effective compression over historical context.

The main questions are:

- Is there a linearly decodable task feature vector in the model's hidden state?
- Do these features correspond to the true sufficient statistics required by the task?
- Do different architectures preserve, compress, or discard prefix information in different ways?
- Do Transformers and Mamba models exhibit different compression behavior because they are not forced to compress all past information into a single recurrent state?
- When moving from controlled synthetic settings to more realistic settings, such as needle-in-a-haystack retrieval, do the internal representations still show evidence of effective feature extraction and compression?

## 2. Main Hypotheses

### 2.1 Sufficient Statistics Hypothesis

If a model has genuinely learned a task, then its hidden state should contain the key statistics needed to solve that task. Examples include:

- Dyck task: the number of unmatched left brackets, stack depth, and the bracket type at the top of the stack.
- Markov task: the current latent state or the posterior distribution over states.
- Needle-in-a-haystack task: whether the needle has appeared, the needle position, and the query-relevant key-value information.

These statistics do not need to appear as explicit coordinates. However, they should be recoverable from some direction or subspace of the hidden state.

### 2.2 Compression Hypothesis

Effective compression is not simply information loss. Rather, it means discarding information that is irrelevant for future prediction while preserving task-relevant sufficient statistics.

This suggests a distinction between two kinds of information:

- Task-relevant information: the statistics required for prediction.
- Task-irrelevant information: irrelevant tokens, noise, positional details, or distractors from the raw prefix.

If a model performs effective compression, its hidden state should preserve task-relevant information with high fidelity while retaining relatively little task-irrelevant information.

## 3. Model Scope

The experiments will compare four classes of architectures:

- RNN: a basic recurrent baseline where a single hidden state is forced to compress history.
- LSTM: a gated recurrent baseline that is, in principle, better suited for long-term information retention.
- Transformer: an attention-based architecture that can directly access contextual tokens and does not need to compress all history into a fixed-size recurrent state.
- Mamba: a state-space / selective-scan architecture, positioned between recurrent compression and long-context modeling.

For a fair comparison, the following variables should be controlled as much as possible:

- Similar number of model parameters.
- Matched embedding dimension, hidden dimension, and number of layers.
- Same tokenizer or synthetic vocabulary.
- Same training data, number of training steps, batch size, optimizer, and evaluation split.

Extract hidden states from the trained model: 
- For RNN, use `state_kind = "h"` and the last layer.
- For LSTM, use `state_kind = "c"` and the last layer.
- For Transformer, use `state_kind = "h"` and the last layer.
- For Mamba, use `state_kind = "h"` and the last layer.

### 3.1 Proposed Model Sizes for Synthetic Tasks

The primary experiments should use small models so that training, hidden-state extraction, probing, and multiple random seeds are all feasible. The exact parameter count will depend on vocabulary size and implementation details, but the following configuration can serve as the default starting point.

| Model | Layers | Embedding / Model Dim | Hidden / State Dim | Other Settings | Approx. Parameters |
|---|---:|---:|---:|---|---:|
| RNN | 3 | 128 | 256 | tanh RNN, causal next-token prediction | 0.25M |
| LSTM | 3 | 128 | 128 | standard LSTM, causal next-token prediction | 0.30M |
| Transformer | 3 | 128 | 128 | 4 attention heads, FFN dim 512, causal mask | 0.45M |
| Mamba | 3 | 128 | 128 | state dim 16, expansion factor 2 | 0.30M-0.50M |

The main comparisons should report both task performance and probing results at matched training conditions. If parameter count becomes a concern, an additional parameter-matched sweep can be run as a control.

### 3.2 Proposed Model Sizes for Realistic Tasks

To be designed.

## 4. Synthetic Pipeline

The goal of the synthetic tasks is to build fully controlled environments where sufficient statistics can be precisely defined and computed.

### 4.1 Dyck and Shuffle Dyck

The task is a causal next-token prediction problem on sequences where Dyck bracket tokens are planted into a longer context. The main object of interest is whether the model's hidden state represents the information needed to predict the Dyck token, especially the prefix structure of the bracket sequence, like the counter of the brackets and height.

#### 4.1.1 Dyck

##### Dyck with 50% Noise

Setting:

- Dyck pairs: `10`.
- Total Length: `20`.
- Sequence length: `48`.
- Training steps: `10000`.
- Repeat probability: `repeat_prob = 0.5`.
- Number of tasks: `n_tasks = 512`.
- Prefix probe max length: `7`.

Evaluation:

- For task accuracy, compute Dyck-only next-token accuracy only on positions where the target token is Dyck-planted.

Prefix probe:

- Extract the hidden representation after a fixed Dyck prefix.
- Compute simple prefix labels from the bracket prefix:
  - `left`: number of `(` tokens.
  - `right`: number of `)` tokens.
  - `height = left - right`.
  - `(left, right)` as an exact prefix-count class.
- Fit linear probes on the hidden states:
  - Ridge regression for `left`, `right`, and `height`, evaluated by `R^2` and MAE.
  - Logistic regression for `height` class and `(left, right)` class, evaluated by accuracy.
- Balance the probe data by `height` so that the learned direction is not dominated by common prefix heights.
- Visualize the learned Ridge directions by projecting hidden states onto the `left`, `right`, and `height` vectors.
- Use PCA only as a sanity check; the main evidence for a counting feature comes from supervised linear probes and their learned directions.
- Sweep over multiple prefix lengths to test whether the counting feature is stable rather than specific to one prefix position.
- Check direction geometry: since `height = left - right`, the learned `height` direction should align with `w_left - w_right`.

##### Dyck with No Noise

Setting:

- Dyck pairs: `24`.
- Total Length: `48`.
- Sequence length: `48`.
- Training steps: `10000`.
- Repeat probability: `repeat_prob = 1`.
- Number of minor tasks: `n_tasks = 512`.
- Prefix probe max length: `7`.

#### 4.1.2 Shuffle Dyck

The task is also causal next-token prediction, but the bracket structure is generated from several interleaved Dyck-like streams. Compared with standard Dyck, the main object of interest is whether the model represents a vector of counters for different bracket types, rather than only a single stack height. The total context is an interleaving of these bracket streams. A closing bracket is valid only when the corresponding type has positive unmatched count.

Setting:

- Use multiple bracket types, for example `()`, `[]`, and `{}`.
- Dyck pairs: `24`.
- Total Length: `48`.
- Sequence length: `48`.
- Training steps: `10000`.
- Repeat probability: `repeat_prob = 1`.
- Number of minor tasks: `n_tasks = 512`.
- Prefix probe max length: `7`.
- Run both no-noise and `repeat_prob = 0.5` versions, matching the Dyck setting.

Evaluation:

- Each bracket type has its own count:
  - `height_round = num_left_round - num_right_round`.
  - `height_square = num_left_square - num_right_square`.
  - `height_curly = num_left_curly - num_right_curly`.
- Compute next-token accuracy on Shuffle-Dyck bracket positions.
- Compare whether the model generalizes to fresh interleavings and longer or more diverse count configurations.

Prefix probe:

- Extract the hidden representation after a fixed Shuffle-Dyck prefix.
- Compute prefix labels from the bracket prefix:
  - Per-type left counts.
  - Per-type right counts.
  - Per-type heights.
  - Total height across all types.
  - The vector of per-type heights as a joint class.
- Fit Ridge probes for scalar count variables, such as each per-type height and total height.
- Fit Logistic probes for discrete labels, such as the joint count-vector class or legal next bracket type.
- Balance the probe data by the count-vector or total height when needed.
- Check whether the hidden state contains separate linear directions for different bracket-type counters.
- Compare with standard Dyck to test whether the model learns a scalar counting feature or a higher-dimensional counting subspace.

#### 4.1.3 Dyck-k

- How to define the Stack?

### 4.2 Resettable Modular Counter

To be designed.

### 4.3 HMM and Variable-order Markov Chains

To be designed.

## 5. Realistic Pipeline

The realistic pipeline should gradually move from controlled symbolic tasks toward long-context retrieval and naturalistic sequence settings.

### 5.1 Needle in a Haystack

To be designed.

### 5.2 Code Generation

To be designed.

### 5.3 Text Analysis with Constraint

To be designed.

## 6. Compression Logistics

### 6.1 Task-Relevant Probe

For each model, layer, and position, train probes on the hidden states:

- Ridge regression: predict continuous statistics such as depth, frequency, or posterior mean.
- Logistic regression: predict discrete statistics such as stack-top type, Markov state, or whether the needle has been found.
- Multiclass classifier: predict state categories, bracket types, or other categorical task variables.

If the probe performs well, this suggests that the hidden state contains a decodable task-relevant feature.

### 6.2 Task-Irrelevant Probe

At the same time, train probes to predict historical details that are irrelevant to the task:

- The identity of a distant token.
- The exact pattern of irrelevant noise tokens in the prefix.
- The content of distractor key-value pairs.
- Randomly inserted irrelevant markers.

If the model retains a large amount of irrelevant information, this indicates weaker compression.

How to evaluate compression?
- Hidden conditional variance
- 

### 6.3 Compression Score

Report two types of metrics separately:

- Relevant retention: the extent to which the model preserves task statistics.
- Irrelevant forgetting: the extent to which the model discards irrelevant historical details.

Effective compression should correspond to high relevant retention and high irrelevant forgetting.
