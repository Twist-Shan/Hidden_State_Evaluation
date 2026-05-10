from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DyckConfig:
    dyck_pairs: int = 24
    total_length: int = 48
    seq_len: int = 48
    repeat_prob: float = 1.0
    n_tasks: int = 512
    prefix_probe_max_len: int = 7
    num_noise_tokens: int = 10
    device: str = "cpu"

    @property
    def close_token(self) -> int:
        return self.num_noise_tokens

    @property
    def open_token(self) -> int:
        return self.num_noise_tokens + 1

    @property
    def vocab_size(self) -> int:
        return self.num_noise_tokens + 2

    def __post_init__(self) -> None:
        if self.total_length > self.seq_len:
            raise ValueError("total_length cannot exceed seq_len")
        if self.total_length % 2 != 0:
            raise ValueError("total_length must be even for a balanced Dyck path")
        if self.dyck_pairs * 2 < self.total_length:
            raise ValueError("dyck_pairs must provide at least total_length / 2 pairs")
