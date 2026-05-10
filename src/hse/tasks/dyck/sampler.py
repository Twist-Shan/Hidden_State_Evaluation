from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import DyckConfig


@dataclass
class DyckBatch:
    tokens: torch.Tensor
    dyck_mask: torch.Tensor
    dyck_steps: torch.Tensor
    noise_tokens: torch.Tensor


class DyckSampler:
    """Generate Dyck paths planted into noise sequences.

    Noise tokens occupy ids `[0, num_noise_tokens)`. The close bracket is
    `num_noise_tokens`, and the open bracket is `num_noise_tokens + 1`.
    """

    def __init__(self, config: DyckConfig, seed: int | None = None):
        self.config = config
        self.device = torch.device(config.device)
        self.generator = torch.Generator(device=self.device)
        if seed is not None:
            self.generator.manual_seed(int(seed))

    @property
    def vocab_size(self) -> int:
        return self.config.vocab_size

    def sample(self, batch_size: int) -> DyckBatch:
        cfg = self.config
        tokens = torch.randint(
            low=0,
            high=cfg.num_noise_tokens,
            size=(batch_size, cfg.seq_len),
            generator=self.generator,
            device=self.device,
        )
        noise_tokens = tokens.clone()
        dyck_steps = self._sample_balanced_steps(batch_size, cfg.total_length)
        positions = self._sample_plant_positions(batch_size)
        dyck_mask = torch.zeros(batch_size, cfg.seq_len, dtype=torch.bool, device=self.device)

        rows = torch.arange(batch_size, device=self.device).unsqueeze(1)
        dyck_mask[rows, positions] = True
        tokens[rows, positions] = torch.where(
            dyck_steps > 0,
            torch.full_like(dyck_steps, cfg.open_token, dtype=torch.long),
            torch.full_like(dyck_steps, cfg.close_token, dtype=torch.long),
        )
        return DyckBatch(tokens=tokens, dyck_mask=dyck_mask, dyck_steps=dyck_steps, noise_tokens=noise_tokens)

    def _sample_plant_positions(self, batch_size: int) -> torch.Tensor:
        cfg = self.config
        if cfg.repeat_prob >= 1.0:
            base = torch.arange(cfg.total_length, device=self.device)
            return base.unsqueeze(0).expand(batch_size, -1)

        positions = []
        all_pos = torch.arange(cfg.seq_len, device=self.device)
        for _ in range(batch_size):
            mask = torch.rand(cfg.seq_len, generator=self.generator, device=self.device) < cfg.repeat_prob
            chosen = all_pos[mask]
            if chosen.numel() < cfg.total_length:
                remaining = all_pos[~mask]
                perm = remaining[torch.randperm(remaining.numel(), generator=self.generator, device=self.device)]
                chosen = torch.cat([chosen, perm[: cfg.total_length - chosen.numel()]])
            elif chosen.numel() > cfg.total_length:
                perm = torch.randperm(chosen.numel(), generator=self.generator, device=self.device)
                chosen = chosen[perm[: cfg.total_length]]
            positions.append(chosen.sort().values)
        return torch.stack(positions, dim=0)

    def _sample_balanced_steps(self, batch_size: int, length: int) -> torch.Tensor:
        path = torch.empty(batch_size, length, dtype=torch.long, device=self.device)
        height = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        opens_used = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        max_opens = length // 2

        for t in range(length):
            remaining = length - t
            must_close = opens_used >= max_opens
            must_open = height <= 0
            can_random = ~(must_close | must_open)
            random_open = torch.rand(batch_size, generator=self.generator, device=self.device) < 0.5
            step_is_open = torch.where(can_random, random_open, must_open)
            # If remaining slots equal current height, all remaining steps must close.
            step_is_open = torch.where(remaining <= height, torch.zeros_like(step_is_open), step_is_open)

            step = torch.where(step_is_open, torch.ones_like(height), -torch.ones_like(height))
            path[:, t] = step
            height += step
            opens_used += step_is_open.long()
        return path
