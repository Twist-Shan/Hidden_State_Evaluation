from __future__ import annotations

import pandas as pd
import torch

from .config import ShuffleDyckConfig
from .sampler import ShuffleDyckBatch


def build_prefix_labels(
    batch: ShuffleDyckBatch,
    config: ShuffleDyckConfig,
    max_prefix_len: int | None = None,
) -> pd.DataFrame:
    """Build one probe-label row per example and sequence position."""
    tokens = batch.tokens.detach().cpu()
    dyck_mask = batch.dyck_mask.detach().cpu()
    noise_tokens = batch.noise_tokens.detach().cpu()
    open_tokens = list(config.open_tokens)
    close_tokens = list(config.close_tokens)
    type_names = list(config.type_names)
    open_to_type = {tok: idx for idx, tok in enumerate(open_tokens)}
    close_to_type = {tok: idx for idx, tok in enumerate(close_tokens)}
    base = config.pairs_per_type + 1

    rows = []
    batch_size, seq_len = tokens.shape
    for example_id in range(batch_size):
        left = [0] * config.num_bracket_types
        right = [0] * config.num_bracket_types
        dyck_seen = 0
        noise_seen = []
        for position in range(seq_len - 1):
            tok = int(tokens[example_id, position])
            is_dyck = bool(dyck_mask[example_id, position])
            if is_dyck:
                dyck_seen += 1
                if tok in open_to_type:
                    left[open_to_type[tok]] += 1
                elif tok in close_to_type:
                    right[close_to_type[tok]] += 1
            else:
                noise_seen.append(int(noise_tokens[example_id, position]))

            if max_prefix_len is not None and dyck_seen > max_prefix_len:
                continue

            heights = [l - r for l, r in zip(left, right)]
            remaining_opens = [config.pairs_per_type - l for l in left]
            legal_next = _encode_legal_next(remaining_opens, heights)

            row = {
                "example_id": example_id,
                "position": position,
                "token": tok,
                "is_dyck_position": is_dyck,
                "dyck_seen": dyck_seen,
                "total_left": sum(left),
                "total_right": sum(right),
                "total_height": sum(heights),
                "count_vector_class": _encode_vector(heights, base=base),
                "count_vector": ",".join(str(v) for v in heights),
                "legal_next_type": legal_next,
                "legal_next_type_class": legal_next,
                "legal_next_count": sum(int(rem > 0) + int(height > 0) for rem, height in zip(remaining_opens, heights)),
                "last_noise_token_class": noise_seen[-1] if noise_seen else -1,
                "distant_noise_token_class": noise_seen[-4] if len(noise_seen) >= 4 else -1,
                "noise_pattern_hash_class": _noise_pattern_hash(noise_seen[-6:], config.num_noise_tokens),
                "random_marker_class": int((example_id * 31 + position * 17) % 8),
            }

            for idx, type_name in enumerate(type_names):
                row[f"left_{type_name}"] = left[idx]
                row[f"right_{type_name}"] = right[idx]
                row[f"height_{type_name}"] = heights[idx]
                row[f"remaining_opens_{type_name}"] = remaining_opens[idx]
            rows.append(row)

    return pd.DataFrame(rows)


def label_row_mask(labels: pd.DataFrame, *, max_prefix_len: int | None = None) -> torch.Tensor:
    mask = torch.ones(len(labels), dtype=torch.bool)
    if max_prefix_len is not None:
        mask &= torch.tensor(labels["dyck_seen"].to_numpy() <= max_prefix_len)
    return mask


def _encode_vector(values: list[int], *, base: int) -> int:
    value = 0
    scale = 1
    for item in values:
        value += int(item) * scale
        scale *= base
    return value


def _encode_legal_next(remaining_opens: list[int], heights: list[int]) -> int:
    mask = 0
    for idx, (rem, height) in enumerate(zip(remaining_opens, heights)):
        if rem > 0:
            mask |= 1 << (2 * idx)
        if height > 0:
            mask |= 1 << (2 * idx + 1)
    return mask


def _noise_pattern_hash(tokens: list[int], num_noise_tokens: int) -> int:
    value = 0
    for tok in tokens:
        value = (value * max(num_noise_tokens, 1) + max(tok, 0)) % 997
    return value
