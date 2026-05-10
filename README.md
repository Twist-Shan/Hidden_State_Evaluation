# Hidden State Evaluation

This project studies whether model hidden states contain linearly decodable sufficient statistics for sequence tasks, and whether different architectures compress historical context in different ways.

The experimental layout follows the style of `mini-ICL-main`: task modules live under `src/`, scripts provide reproducible entry points, notebooks are used for exploration, and `results/` stores generated checkpoints, probes, metrics, and figures.

## Research Questions

- Do RNN, LSTM, Transformer, and Mamba hidden states contain task-relevant sufficient statistics?
- Are those statistics linearly decodable from a hidden direction or subspace?
- Do models preserve task-relevant information while discarding irrelevant prefix details?
- How does compression behavior differ between recurrent models, attention models, and state-space models?

## Task Families

| Task | Module | Current Role |
|---|---|---|
| Dyck | `hse.tasks.dyck` | Main controlled counting task |
| Shuffle Dyck | `hse.tasks.shuffle_dyck` | Multi-counter bracket task |
| Resettable Counter | `hse.tasks.markov` | Placeholder for finite-state sufficient statistics |
| HMM / Markov | `hse.tasks.markov` | Placeholder for latent-state and posterior probes |
| Needle in a Haystack | `hse.tasks.needle` | Placeholder for realistic retrieval/compression tests |

## Project Structure

```text
Hidden State Evaluation/
|-- src/hse/
|   |-- models/                  # RNN, LSTM, Transformer, Mamba model wrappers
|   |-- tasks/
|   |   |-- dyck/                # Dyck samplers, labels, metrics
|   |   |-- shuffle_dyck/        # Interleaved multi-bracket task
|   |   |-- markov/              # Counter, HMM, variable-order Markov tasks
|   |   `-- needle/              # Needle-in-a-haystack retrieval tasks
|   |-- analysis/
|   |   |-- probes/              # Ridge/logistic probes for hidden states
|   |   |-- geometry/            # Direction alignment, PCA, subspace analysis
|   |   |-- compression/         # Relevant retention and irrelevant forgetting scores
|   |   `-- visualization/       # Probe plots and trajectory figures
|   `-- utils/                   # Training, extraction, config, IO helpers
|-- configs/                     # Reproducible experiment configs
|-- scripts/                     # Command-line experiment entry points
|-- notebooks/                   # Exploratory notebooks
|-- docs/                        # Design notes and experiment logs
|-- tests/                       # Unit tests for samplers, labels, probes
|-- results/                     # Generated outputs, checkpoints, metrics
|-- paper_figs/                  # Final exported figures
`-- experiment_pipeline_plan.md  # Original research plan
```

## Default Synthetic Experiments

The first implementation target is Dyck and Shuffle Dyck with matched model sizes.

| Model | Layers | Embedding Dim | Hidden / State Dim | Notes |
|---|---:|---:|---:|---|
| RNN | 3 | 128 | 256 | tanh recurrent baseline |
| LSTM | 3 | 128 | 128 | use cell state `c` for extraction |
| Transformer | 3 | 128 | 128 | causal attention, 4 heads |
| Mamba | 3 | 128 | 128 | state dim 16, expansion 2 |

### Dyck With Noise

- Dyck pairs: `10`
- Total Dyck length: `20`
- Sequence length: `48`
- Training steps: `10000`
- Repeat probability: `0.5`
- Number of tasks: `512`
- Prefix probe max length: `7`

### Dyck Without Noise

- Dyck pairs: `24`
- Total Dyck length: `48`
- Sequence length: `48`
- Training steps: `10000`
- Repeat probability: `1.0`
- Number of tasks: `512`
- Prefix probe max length: `7`

### Shuffle Dyck

- Bracket types: `()`, `[]`, `{}`
- Dyck pairs: `24`
- Total length: `48`
- Sequence length: `48`
- Run both `repeat_prob = 1.0` and `repeat_prob = 0.5`

## Planned Workflow

1. Train matched models on a controlled task.
2. Evaluate task accuracy on task-relevant token positions.
3. Extract hidden states for each architecture:
   - RNN: `state_kind = "h"`, last layer.
   - LSTM: `state_kind = "c"`, last layer.
   - Transformer: `state_kind = "h"`, last layer.
   - Mamba: `state_kind = "h"`, last layer.
4. Build prefix labels such as `left`, `right`, `height`, per-type counters, legal next-token class, and irrelevant prefix details.
5. Fit Ridge and logistic probes.
6. Analyze learned probe directions:
   - `height` should align with `left - right`.
   - Shuffle Dyck should produce separate directions for separate bracket counters.
7. Report compression using relevant retention and irrelevant forgetting metrics.

## Script Entry Points

The script files are intentionally thin placeholders for now. They define the stable command interface that future code should implement.

```bash
python scripts/train_model.py --config configs/dyck_noise.yaml
python scripts/extract_hidden_states.py --run results/dyck_noise/rnn_seed0
python scripts/run_probes.py --features results/dyck_noise/rnn_seed0/hidden_states.pt
python scripts/analyze_geometry.py --probe-dir results/dyck_noise/rnn_seed0/probes
python scripts/run_pipeline.py --config configs/dyck_noise.yaml
```

## Output Convention

```text
results/
`-- dyck_noise/
    `-- rnn_seed0/
        |-- config.yaml
        |-- checkpoints/
        |-- metrics.json
        |-- hidden_states.pt
        |-- labels.parquet
        |-- probes/
        `-- figures/
```

## Chinese Notes

这个文件夹现在是实验架构骨架，还不是完整可运行实现。它的设计目标是把 `experiment_pipeline_plan.md` 里的研究问题落成清晰的工程边界：

- `tasks` 负责生成序列和真实 sufficient statistics 标签。
- `models` 负责不同架构的统一封装。
- `utils` 负责训练、保存、加载、hidden state extraction。
- `analysis/probes` 负责线性探针。
- `analysis/geometry` 负责方向几何和子空间分析。
- `analysis/compression` 负责区分 task-relevant retention 和 task-irrelevant forgetting。

后续真正写实验代码时，优先实现 Dyck no-noise 和 Dyck 50% noise 两条线，再扩展到 Shuffle Dyck、Markov/HMM 和 Needle-in-a-Haystack。
