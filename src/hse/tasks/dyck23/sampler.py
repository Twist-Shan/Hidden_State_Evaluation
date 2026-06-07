from __future__ import annotations

from dataclasses import dataclass
import random

import torch

from .config import Dyck23Config


@dataclass
class Dyck23Batch:
    tokens: torch.Tensor
    dyck_mask: torch.Tensor
    target_mask: torch.Tensor
    lengths: torch.Tensor
    bracket_lengths: torch.Tensor
    dyck_steps: torch.Tensor
    bracket_type_ids: torch.Tensor


class Dyck23Sampler:
    """Sample true Dyck-(2,3) strings from a bounded-depth CFG.

    Grammar by current depth d:

    S_d -> epsilon | B_d S_d
    B_d -> '(' S_{d+1} ')' | '[' S_{d+1} ']'

    The nonterminal S_max_depth only derives epsilon, which enforces the
    maximum nesting depth exactly at generation time.
    """

    def __init__(self, config: Dyck23Config, seed: int | None = None):
        self.config = config
        self.device = torch.device(config.device)
        self.rng = random.Random(seed)
        self._counts = self._build_counts(config.max_length // 2)

    @property
    def vocab_size(self) -> int:
        return self.config.vocab_size

    def sample(self, batch_size: int) -> Dyck23Batch:
        cfg = self.config
        seq_len = cfg.max_sequence_length
        tokens = torch.full((batch_size, seq_len), cfg.pad_token, dtype=torch.long)
        dyck_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        target_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        dyck_steps = torch.zeros(batch_size, seq_len, dtype=torch.long)
        bracket_type_ids = torch.full((batch_size, seq_len), -1, dtype=torch.long)
        lengths = torch.zeros(batch_size, dtype=torch.long)
        bracket_lengths = torch.zeros(batch_size, dtype=torch.long)

        for row in range(batch_size):
            bracket_length = self.rng.choice(cfg.valid_lengths)
            bracket_tokens, steps, type_ids = self._sample_string(bracket_length // 2)
            payload_length = cfg.payload_length_for(bracket_length)
            payload = self._sample_noise_payload(payload_length)
            positions = self._sample_plant_positions(bracket_length, payload_length)
            for pos, tok, step, type_id in zip(positions, bracket_tokens, steps, type_ids):
                payload[pos] = tok
                seq_pos = 1 + pos
                dyck_mask[row, seq_pos] = True
                dyck_steps[row, seq_pos] = step
                bracket_type_ids[row, seq_pos] = type_id

            sequence = [cfg.bos_token, *payload, cfg.eos_token]
            length = len(sequence)

            tokens[row, :length] = torch.tensor(sequence, dtype=torch.long)
            target_mask[row, 1:length] = True
            lengths[row] = length
            bracket_lengths[row] = bracket_length

        return Dyck23Batch(
            tokens=tokens.to(self.device),
            dyck_mask=dyck_mask.to(self.device),
            target_mask=target_mask.to(self.device),
            lengths=lengths.to(self.device),
            bracket_lengths=bracket_lengths.to(self.device),
            dyck_steps=dyck_steps.to(self.device),
            bracket_type_ids=bracket_type_ids.to(self.device),
        )

    def _sample_noise_payload(self, payload_length: int) -> list[int]:
        cfg = self.config
        if cfg.num_noise_tokens == 0:
            return [cfg.pad_token] * payload_length
        return [self.rng.choice(cfg.noise_tokens) for _ in range(payload_length)]

    def _sample_plant_positions(self, bracket_length: int, payload_length: int) -> list[int]:
        cfg = self.config
        if bracket_length == 0:
            return []
        if cfg.generation_prob >= 1.0:
            return list(range(bracket_length))

        all_positions = list(range(payload_length))
        chosen = [pos for pos in all_positions if self.rng.random() < cfg.generation_prob]
        if len(chosen) < bracket_length:
            chosen_set = set(chosen)
            remaining = [pos for pos in all_positions if pos not in chosen_set]
            chosen.extend(self.rng.sample(remaining, bracket_length - len(chosen)))
        elif len(chosen) > bracket_length:
            chosen = self.rng.sample(chosen, bracket_length)
        return sorted(chosen)

    def _build_counts(self, max_pairs: int) -> list[list[int]]:
        cfg = self.config
        counts = [[0 for _ in range(max_pairs + 1)] for _ in range(cfg.max_depth + 1)]
        counts[cfg.max_depth][0] = 1

        for depth in range(cfg.max_depth - 1, -1, -1):
            counts[depth][0] = 1
            for pairs in range(1, max_pairs + 1):
                total = 0
                for first_pairs in range(1, pairs + 1):
                    inside_pairs = first_pairs - 1
                    rest_pairs = pairs - first_pairs
                    total += (
                        len(cfg.bracket_types)
                        * counts[depth + 1][inside_pairs]
                        * counts[depth][rest_pairs]
                    )
                counts[depth][pairs] = total
        return counts

    def _sample_string(self, pairs: int, depth: int = 0) -> tuple[list[int], list[int], list[int]]:
        cfg = self.config
        if pairs == 0:
            return [], [], []
        if depth >= cfg.max_depth:
            raise ValueError("Cannot derive a non-empty string at max depth")

        weights = []
        for first_pairs in range(1, pairs + 1):
            inside_pairs = first_pairs - 1
            rest_pairs = pairs - first_pairs
            weights.append(
                len(cfg.bracket_types)
                * self._counts[depth + 1][inside_pairs]
                * self._counts[depth][rest_pairs]
            )
        total = sum(weights)
        pick = self.rng.randrange(total)
        first_pairs = 1
        for idx, weight in enumerate(weights, start=1):
            if pick < weight:
                first_pairs = idx
                break
            pick -= weight

        inside_pairs = first_pairs - 1
        rest_pairs = pairs - first_pairs
        type_id = self.rng.randrange(len(cfg.bracket_types))
        inside_tokens, inside_steps, inside_types = self._sample_string(inside_pairs, depth + 1)
        rest_tokens, rest_steps, rest_types = self._sample_string(rest_pairs, depth)

        open_tok = cfg.open_tokens[type_id]
        close_tok = cfg.close_tokens[type_id]
        return (
            [open_tok, *inside_tokens, close_tok, *rest_tokens],
            [1, *inside_steps, -1, *rest_steps],
            [type_id, *inside_types, type_id, *rest_types],
        )


def validate_dyck23_tokens(tokens: list[int], config: Dyck23Config) -> bool:
    """Return True when tokens form a valid Dyck-(2,3) bracket-only string."""
    open_to_type = {tok: idx for idx, tok in enumerate(config.open_tokens)}
    close_to_type = {tok: idx for idx, tok in enumerate(config.close_tokens)}
    stack: list[int] = []
    for tok in tokens:
        if tok in open_to_type:
            stack.append(open_to_type[tok])
            if len(stack) > config.max_depth:
                return False
        elif tok in close_to_type:
            if not stack or stack[-1] != close_to_type[tok]:
                return False
            stack.pop()
        else:
            return False
    return not stack
