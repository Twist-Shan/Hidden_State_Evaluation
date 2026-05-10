from __future__ import annotations

import pandas as pd
import torch

from .config import DyckConfig
from .sampler import DyckBatch


def build_prefix_labels(batch: DyckBatch, config: DyckConfig | None = None, max_prefix_len: int | None = None) -> pd.DataFrame:
    """Build one probe-label row per example and sequence position."""
    tokens = batch.tokens.detach().cpu()
    dyck_mask = batch.dyck_mask.detach().cpu()
    noise_tokens = batch.noise_tokens.detach().cpu()

    if config is None:
        num_noise = int(tokens.max().item()) - 1
        open_token = num_noise + 1
        close_token = num_noise
        dyck_total = int(dyck_mask.sum(dim=1).max().item())
    else:
        num_noise = config.num_noise_tokens
        open_token = config.open_token
        close_token = config.close_token
        dyck_total = config.total_length

    rows = []
    batch_size, seq_len = tokens.shape
    for example_id in range(batch_size):
        left = 0
        right = 0
        dyck_seen = 0
        noise_seen = []
        for position in range(seq_len - 1):
            tok = int(tokens[example_id, position])
            is_dyck = bool(dyck_mask[example_id, position])
            if is_dyck:
                dyck_seen += 1
                if tok == open_token:
                    left += 1
                elif tok == close_token:
                    right += 1
            else:
                noise_seen.append(int(noise_tokens[example_id, position]))

            if max_prefix_len is not None and dyck_seen > max_prefix_len:
                continue

            height = left - right
            remaining_opens = dyck_total // 2 - left
            can_open = remaining_opens > 0
            can_close = height > 0
            if can_open and can_close:
                legal_next_class = 2
            elif can_close:
                legal_next_class = 1
            else:
                legal_next_class = 0

            last_noise = noise_seen[-1] if noise_seen else -1
            distant_noise = noise_seen[-4] if len(noise_seen) >= 4 else -1
            pattern_hash = _noise_pattern_hash(noise_seen[-6:], num_noise)

            rows.append(
                {
                    "example_id": example_id,
                    "position": position,
                    "token": tok,
                    "is_dyck_position": is_dyck,
                    "dyck_seen": dyck_seen,
                    "left": left,
                    "right": right,
                    "height": height,
                    "height_class": height,
                    "left_right_class": left * (dyck_total + 1) + right,
                    "legal_next_class": legal_next_class,
                    "last_noise_token_class": last_noise,
                    "distant_noise_token_class": distant_noise,
                    "noise_pattern_hash_class": pattern_hash,
                    "random_marker_class": int((example_id * 31 + position * 17) % 8),
                }
            )
    return pd.DataFrame(rows)


def label_row_mask(labels: pd.DataFrame, *, max_prefix_len: int | None = None) -> torch.Tensor:
    mask = torch.ones(len(labels), dtype=torch.bool)
    if max_prefix_len is not None:
        mask &= torch.tensor(labels["dyck_seen"].to_numpy() <= max_prefix_len)
    return mask


def _noise_pattern_hash(tokens: list[int], num_noise_tokens: int) -> int:
    value = 0
    for tok in tokens:
        value = (value * max(num_noise_tokens, 1) + max(tok, 0)) % 997
    return value
