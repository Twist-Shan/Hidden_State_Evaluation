from __future__ import annotations

import pandas as pd
import torch

from .config import DyckKConfig
from .sampler import DyckKBatch


def build_prefix_labels(
    batch: DyckKBatch,
    config: DyckKConfig,
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
    vector_base = config.total_pairs + 1

    rows = []
    batch_size, seq_len = tokens.shape
    for example_id in range(batch_size):
        left = [0] * config.num_bracket_types
        right = [0] * config.num_bracket_types
        stack: list[int] = []
        dyck_seen = 0
        noise_seen = []

        for position in range(seq_len - 1):
            tok = int(tokens[example_id, position])
            is_dyck = bool(dyck_mask[example_id, position])
            if is_dyck:
                dyck_seen += 1
                if tok in open_to_type:
                    type_id = open_to_type[tok]
                    left[type_id] += 1
                    stack.append(type_id)
                elif tok in close_to_type:
                    type_id = close_to_type[tok]
                    right[type_id] += 1
                    if stack and stack[-1] == type_id:
                        stack.pop()
            else:
                noise_seen.append(int(noise_tokens[example_id, position]))

            if max_prefix_len is not None and dyck_seen > max_prefix_len:
                continue

            heights = [l - r for l, r in zip(left, right)]
            depth = len(stack)
            top_type = stack[-1] + 1 if stack else 0
            legal_next_close_token = close_tokens[stack[-1]] if stack else -1

            row = {
                "example_id": example_id,
                "position": position,
                "token": tok,
                "is_dyck_position": is_dyck,
                "dyck_seen": dyck_seen,
                "depth": depth,
                "top_type_class": top_type,
                "top_type_name": "empty" if not stack else type_names[stack[-1]],
                "top_2_class": _encode_stack_suffix(stack, type_names=type_names, depth=2),
                "top_3_class": _encode_stack_suffix(stack, type_names=type_names, depth=3),
                "stack_repr": _stack_repr(stack, type_names=type_names),
                "height_vector_class": _encode_vector(heights, base=vector_base),
                "height_vector": ",".join(str(v) for v in heights),
                "legal_next_close_type": top_type,
                "legal_next_close_name": "none" if not stack else type_names[stack[-1]],
                "legal_next_close_token": legal_next_close_token,
                "depth_top_class": depth * (config.num_bracket_types + 1) + top_type,
                "remaining_total_opens": config.total_pairs - sum(left),
                "last_noise_token_class": noise_seen[-1] if noise_seen else -1,
                "distant_noise_token_class": noise_seen[-4] if len(noise_seen) >= 4 else -1,
                "noise_pattern_hash_class": _noise_pattern_hash(noise_seen[-6:], config.num_noise_tokens),
                "random_marker_class": int((example_id * 31 + position * 17) % 8),
            }

            for idx, type_name in enumerate(type_names):
                row[f"left_{type_name}"] = left[idx]
                row[f"right_{type_name}"] = right[idx]
                row[f"height_{type_name}"] = heights[idx]
            rows.append(row)

    return pd.DataFrame(rows)


def label_row_mask(labels: pd.DataFrame, *, max_prefix_len: int | None = None) -> torch.Tensor:
    mask = torch.ones(len(labels), dtype=torch.bool)
    if max_prefix_len is not None:
        mask &= torch.tensor(labels["dyck_seen"].to_numpy() <= max_prefix_len)
    return mask


def _encode_stack_suffix(stack: list[int], *, type_names: list[str], depth: int) -> str:
    pad = ["empty"] * max(depth - len(stack), 0)
    suffix = [type_names[idx] for idx in reversed(stack[-depth:])]
    return "|".join(suffix + pad)


def _stack_repr(stack: list[int], *, type_names: list[str]) -> str:
    if not stack:
        return "empty"
    return "|".join(type_names[idx] for idx in reversed(stack))


def _encode_vector(values: list[int], *, base: int) -> int:
    value = 0
    scale = 1
    for item in values:
        value += int(item) * scale
        scale *= base
    return value


def _noise_pattern_hash(tokens: list[int], num_noise_tokens: int) -> int:
    value = 0
    for tok in tokens:
        value = (value * max(num_noise_tokens, 1) + max(tok, 0)) % 997
    return value
