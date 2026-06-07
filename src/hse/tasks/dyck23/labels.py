from __future__ import annotations

import pandas as pd

from .config import Dyck23Config
from .sampler import Dyck23Batch


def build_prefix_labels(
    batch: Dyck23Batch,
    config: Dyck23Config,
    max_prefix_len: int | None = None,
) -> pd.DataFrame:
    """Build one label row per next-token prediction position."""
    tokens = batch.tokens.detach().cpu()
    dyck_mask = batch.dyck_mask.detach().cpu()
    target_mask = batch.target_mask.detach().cpu()
    bracket_lengths = batch.bracket_lengths.detach().cpu()
    open_tokens = list(config.open_tokens)
    close_tokens = list(config.close_tokens)
    noise_tokens = set(config.noise_tokens)
    type_names = list(config.type_names)
    open_to_type = {tok: idx for idx, tok in enumerate(open_tokens)}
    close_to_type = {tok: idx for idx, tok in enumerate(close_tokens)}

    rows = []
    batch_size, seq_len = tokens.shape
    for example_id in range(batch_size):
        left = [0] * len(open_tokens)
        right = [0] * len(close_tokens)
        stack: list[int] = []
        dyck_seen = 0
        noise_seen: list[int] = []
        bracket_length = int(bracket_lengths[example_id])

        for position in range(seq_len - 1):
            if not bool(target_mask[example_id, position + 1]):
                continue
            tok = int(tokens[example_id, position])
            next_tok = int(tokens[example_id, position + 1])
            is_dyck = bool(dyck_mask[example_id, position])

            if is_dyck and tok in open_to_type:
                type_id = open_to_type[tok]
                left[type_id] += 1
                stack.append(type_id)
                dyck_seen += 1
            elif is_dyck and tok in close_to_type:
                type_id = close_to_type[tok]
                right[type_id] += 1
                if stack and stack[-1] == type_id:
                    stack.pop()
                dyck_seen += 1
            elif tok in noise_tokens:
                noise_seen.append(tok)

            if max_prefix_len is not None and dyck_seen > max_prefix_len:
                continue

            height = len(stack)
            remaining_slots = bracket_length - dyck_seen
            remaining_opens = bracket_length // 2 - sum(left)
            can_open = remaining_slots > height and remaining_opens > 0 and height < config.max_depth
            can_close = height > 0
            can_eos = remaining_slots == 0
            legal_next_class = int(can_open) + 2 * int(can_close) + 4 * int(can_eos)
            top_type = stack[-1] + 1 if stack else 0
            legal_next_close_token = close_tokens[stack[-1]] if stack else -1
            heights = [l - r for l, r in zip(left, right)]

            row = {
                "example_id": example_id,
                "position": position,
                "token": tok,
                "next_token": next_tok,
                "is_dyck_position": is_dyck,
                "bracket_length": bracket_length,
                "dyck_seen": dyck_seen,
                "left": sum(left),
                "right": sum(right),
                "height": height,
                "height_class": height,
                "depth": height,
                "top_type_class": top_type,
                "top_type_name": "empty" if not stack else type_names[stack[-1]],
                "top_2_class": _encode_stack_suffix(stack, type_names=type_names, depth=2),
                "top_3_class": _encode_stack_suffix(stack, type_names=type_names, depth=3),
                "stack_repr": _stack_repr(stack, type_names=type_names),
                "legal_next_class": legal_next_class,
                "legal_next_close_type": top_type,
                "legal_next_close_token": legal_next_close_token,
                "depth_top_class": height * (len(open_tokens) + 1) + top_type,
                "remaining_total_opens": remaining_opens,
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


def _encode_stack_suffix(stack: list[int], *, type_names: list[str], depth: int) -> str:
    pad = ["empty"] * max(depth - len(stack), 0)
    suffix = [type_names[idx] for idx in reversed(stack[-depth:])]
    return "|".join(suffix + pad)


def _stack_repr(stack: list[int], *, type_names: list[str]) -> str:
    if not stack:
        return "empty"
    return "|".join(type_names[idx] for idx in reversed(stack))


def _noise_pattern_hash(tokens: list[int], num_noise_tokens: int) -> int:
    value = 0
    for tok in tokens:
        value = (value * max(num_noise_tokens, 1) + max(tok, 0)) % 997
    return value
