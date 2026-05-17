from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import DyckKConfig


@dataclass
class DyckKBatch:
    tokens: torch.Tensor
    dyck_mask: torch.Tensor
    dyck_steps: torch.Tensor
    noise_tokens: torch.Tensor
    bracket_type_ids: torch.Tensor


class DyckKSampler:
    """Generate valid multi-type Dyck sequences with stack discipline."""

    def __init__(self, config: DyckKConfig, seed: int | None = None):
        self.config = config
        self.device = torch.device(config.device)
        self.generator = torch.Generator(device=self.device)
        if seed is not None:
            self.generator.manual_seed(int(seed))

    @property
    def vocab_size(self) -> int:
        return self.config.vocab_size

    def sample(self, batch_size: int) -> DyckKBatch:
        cfg = self.config
        tokens = torch.randint(
            low=0,
            high=cfg.num_noise_tokens,
            size=(batch_size, cfg.seq_len),
            generator=self.generator,
            device=self.device,
        )
        noise_tokens = tokens.clone()
        dyck_steps, bracket_type_ids = self._sample_stack_paths(batch_size)
        positions = self._sample_plant_positions(batch_size)
        dyck_mask = torch.zeros(batch_size, cfg.seq_len, dtype=torch.bool, device=self.device)

        rows = torch.arange(batch_size, device=self.device).unsqueeze(1)
        dyck_mask[rows, positions] = True

        bracket_tokens = torch.empty(batch_size, cfg.total_length, dtype=torch.long, device=self.device)
        for type_id, (open_tok, close_tok) in enumerate(zip(cfg.open_tokens, cfg.close_tokens)):
            is_type = bracket_type_ids == type_id
            bracket_tokens[is_type] = torch.where(
                dyck_steps[is_type] > 0,
                torch.full_like(dyck_steps[is_type], open_tok),
                torch.full_like(dyck_steps[is_type], close_tok),
            )
        tokens[rows, positions] = bracket_tokens

        return DyckKBatch(
            tokens=tokens,
            dyck_mask=dyck_mask,
            dyck_steps=dyck_steps,
            noise_tokens=noise_tokens,
            bracket_type_ids=bracket_type_ids,
        )

    def _sample_plant_positions(self, batch_size: int) -> torch.Tensor:
        cfg = self.config
        if cfg.generation_prob >= 1.0:
            base = torch.arange(cfg.total_length, device=self.device)
            return base.unsqueeze(0).expand(batch_size, -1)

        positions = []
        all_pos = torch.arange(cfg.seq_len, device=self.device)
        for _ in range(batch_size):
            mask = torch.rand(cfg.seq_len, generator=self.generator, device=self.device) < cfg.generation_prob
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

    def _sample_stack_paths(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        rows = torch.arange(batch_size, device=self.device)
        stack = torch.full((batch_size, cfg.total_pairs), -1, dtype=torch.long, device=self.device)
        depths = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        opens_used = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        dyck_steps = torch.empty(batch_size, cfg.total_length, dtype=torch.long, device=self.device)
        bracket_type_ids = torch.empty_like(dyck_steps)

        for t in range(cfg.total_length):
            remaining = cfg.total_length - t
            can_close = depths > 0
            can_open = (opens_used < cfg.total_pairs) & (remaining > (depths + 1))

            must_open = ~can_close
            must_close = ~can_open
            random_open = torch.rand(batch_size, generator=self.generator, device=self.device) < 0.5
            step_is_open = torch.where(must_open, torch.ones_like(random_open), random_open)
            step_is_open = torch.where(must_close, torch.zeros_like(step_is_open), step_is_open)

            dyck_steps[:, t] = torch.where(
                step_is_open,
                torch.ones_like(depths),
                -torch.ones_like(depths),
            )

            if bool(step_is_open.any()):
                open_rows = rows[step_is_open]
                open_depths = depths[step_is_open]
                open_types = torch.randint(
                    low=0,
                    high=cfg.num_bracket_types,
                    size=(open_rows.numel(),),
                    generator=self.generator,
                    device=self.device,
                )
                stack[open_rows, open_depths] = open_types
                bracket_type_ids[step_is_open, t] = open_types

            close_mask = ~step_is_open
            if bool(close_mask.any()):
                close_rows = rows[close_mask]
                close_depths = depths[close_mask] - 1
                close_types = stack[close_rows, close_depths]
                bracket_type_ids[close_mask, t] = close_types

            depths = depths + dyck_steps[:, t]
            opens_used = opens_used + step_is_open.long()

        return dyck_steps, bracket_type_ids
