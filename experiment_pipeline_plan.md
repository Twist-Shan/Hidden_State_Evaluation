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

### 4.1 Dyck 

The task is a causal next-token prediction problem on sequences where Dyck bracket tokens are planted into a longer context. The main object of interest is whether the model's hidden state represents the information needed to predict the Dyck token, especially the prefix structure of the bracket sequence, like the counter of the brackets and height.

#### 4.1.1 Dyck

##### Dyck with 50% Noise

Setting:

- Dyck pairs: `24`. 
- Total length: `48`. 
- Sequence length: `120`. 
- Repeat probability: `0.5`.
- Training steps: `10000`.
- Batch size: `128`.
- Learning rate: `3e-4`.
- Probe prefix length: `8`.

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
- Total length: `48`.
- Sequence length: `48`.
- Repeat probability: `1.0` (no noise).
- Training steps: `10000`.
- Batch size: `128`.
- Learning rate: `3e-4`.
- Probe prefix length: `8`.

#### 4.1.2 Shuffle Dyck

The task is also causal next-token prediction, but the bracket structure is generated from several interleaved Dyck-like streams. Compared with standard Dyck, the main object of interest is whether the model represents a vector of counters for different bracket types, rather than only a single stack height. The total context is an interleaving of these bracket streams. A closing bracket is valid only when the corresponding type has positive unmatched count.

##### Shuffle Dyck with 50% Noise

Setting:

- Task: Shuffle Dyck.
- Bracket types: `()`, `[]`, `{}`.
- Pairs per bracket type: `8`.
- Total bracket length: `48`.
- Sequence length: `120`.
- Generation probability: `0.5` (no noise).
- Training steps: `10000`.
- Batch size: `128`.
- Learning rate: `3e-4`.
- Probe prefix length: `8`.

##### Shuffle Dyck with No Noise

Setting:

- Task: Shuffle Dyck.
- Bracket types: `()`, `[]`, `{}`.
- Pairs per bracket type: `8`.
- Total bracket length: `48`.
- Sequence length: `48`.
- Generation probability: `1.0` (no noise).
- Training steps: `10000`.
- Batch size: `128`.
- Learning rate: `3e-4`.
- Probe prefix length: `8`.

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

The task is causal next-token prediction on balanced parentheses with multiple bracket types. Unlike Shuffle Dyck, Dyck-\(k\) requires a true stack discipline: a closing bracket is legal only if it matches the most recently opened unmatched bracket. Therefore, the sufficient state is no longer only a vector of per-type counts. The model must track the current stack content, or at least the stack information relevant for predicting the next legal closing bracket.

For example, with two bracket types `()` and `[]`:

- The sequence `([)]` is valid under Shuffle Dyck-style independent counters, because both types have balanced counts.
- But it is invalid under Dyck-\(k\), because after reading `([`, the top of the stack is `[`, so the next closing bracket must be `]`, not `)`.

Thus, Dyck-\(k\) separates multi-counter compression from stack-like compression.

##### Dyck-\(k\) with No Noise

Setting:

- Task: Dyck-\(k\).
- Bracket types: `()`, `[]`, `{}`.
- Total bracket pairs: `24`.
- Total bracket length: `48`.
- Sequence length: `48`.
- Generation probability: `1.0` (no noise).
- Training steps: `10000`.
- Batch size: `128`.
- Learning rate: `3e-4`.
- Probe prefix length: `8`.

Generation:

- Generate a valid Dyck-\(k\) sequence using a stack.
- At each step, if the stack is empty, the next bracket must be an opening bracket.
- If the stack is nonempty, either:
  - open a new bracket of one of the \(k\) types, if the remaining budget allows;
  - or close the current top-of-stack bracket.
- A closing bracket must match the top stack symbol.
- The sequence is valid if the stack is empty at the end and all bracket pairs are used.

Stack definition:

- Let the stack at prefix length \(t\) be
  \[
  S_t = (s_1,s_2,\dots,s_{d_t}),
  \]
  where \(d_t\) is the current depth and \(s_{d_t}\) is the top of the stack.
- Each stack element is a bracket type:
  \[
  s_i \in \{1,2,\dots,k\}.
  \]
- When reading an opening bracket of type \(a\), update
  \[
  S_{t+1} = (S_t,a).
  \]
- When reading a closing bracket of type \(a\), the transition is legal only if
  \[
  \operatorname{top}(S_t)=a.
  \]
  Then
  \[
  S_{t+1}=S_t \text{ with its top element removed}.
  \]

Evaluation:

- Compute next-token accuracy on all bracket positions.
- Separately compute accuracy on:
  - opening-bracket targets;
  - closing-bracket targets;
  - positions where the stack depth is small;
  - positions where the stack depth is large.
- Compute legal-next-token accuracy:
  - whether the model assigns high probability to legal next brackets;
  - whether it correctly identifies the unique legal closing bracket when the next token is a close.
- Evaluate generalization to:
  - fresh bracket-type patterns;
  - longer sequences;
  - deeper nesting than seen during training.

Prefix probe:

- Extract the hidden representation after a fixed Dyck-\(k\) prefix.
- Compute prefix labels from the bracket prefix:
  - `depth`: current stack depth \(d_t\).
  - `top`: current top-of-stack bracket type.
  - `top-2`: the top two stack symbols, if available.
  - `top-r`: the top \(r\) stack symbols, padded if the stack has depth less than \(r\).
  - per-type left counts.
  - per-type right counts.
  - per-type heights.
  - full stack class, for small maximum depth.
- Fit Ridge probes for scalar variables:
  - depth;
  - per-type heights.
- Fit Logistic probes for discrete variables:
  - top-of-stack type;
  - top-2 stack class;
  - top-\(r\) stack class;
  - full stack class when the number of possible stacks is manageable;
  - legal next closing bracket.
- Balance the probe data by stack depth and top-of-stack type so that the probe is not dominated by shallow stacks or frequent bracket types.
- Compare probes for:
  - depth alone;
  - per-type height vector;
  - top-of-stack;
  - top-\(r\) stack symbols.

Main representation questions:

- Does the model encode only the scalar depth \(d_t\), or does it encode the full stack structure?
- Can the model linearly decode the top-of-stack symbol?
- Does the representation contain information about deeper stack elements, such as top-2 or top-3?
- Are prefixes with the same per-type counts but different stack order separated in hidden space?

Geometry analysis:

- Project hidden states using supervised directions for:
  - depth;
  - top-of-stack;
  - per-type heights.
- Use PCA only as a sanity check.
- The main evidence should come from supervised probes and controlled prefix-pair comparisons.
- Compare Dyck-\(k\) against Shuffle Dyck:
  - Shuffle Dyck should be solvable by a vector of counters.
  - Dyck-\(k\) requires stack order information.
- This comparison tests whether the model learns:
  - scalar counting;
  - multi-dimensional counting;
  - or stack-like structured memory.

##### Dyck-\(k\) with 50% Noise

Setting:

- Task: Dyck-\(k\) with noise.
- Bracket types: `()`, `[]`, `{}`.
- Number of bracket types: `3`.
- Total bracket pairs: `10`.
- Total bracket length: `20`.
- Sequence length: `48`.
- Repeat probability: `0.5`.
- Training steps: `10000`.
- Batch size: `128`.
- Learning rate: `3e-4`.
- Probe prefix length: `8`.

Generation:

- First generate a valid Dyck-\(k\) bracket sequence of length `20`.
- Insert noise tokens into non-bracket positions until the total sequence length is `48`.
- Noise tokens do not affect the stack.
- The stack is updated only when a Dyck bracket token appears.

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
