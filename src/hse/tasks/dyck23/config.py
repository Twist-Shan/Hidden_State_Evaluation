from __future__ import annotations

from dataclasses import dataclass
import math


BRACKET_NAME_BY_PAIR = {
    "()": "round",
    "[]": "square",
}


@dataclass
class Dyck23Config:
    """Configuration for CFG-generated Dyck-(2,3) next-token data."""

    min_length: int = 0
    max_length: int = 40
    max_depth: int = 3
    bracket_types: tuple[str, ...] = ("()", "[]")
    generation_prob: float = 1.0
    num_noise_tokens: int = 0
    device: str = "cpu"

    @property
    def pad_token(self) -> int:
        return 0

    @property
    def bos_token(self) -> int:
        return 1

    @property
    def eos_token(self) -> int:
        return 2

    @property
    def noise_tokens(self) -> tuple[int, ...]:
        return tuple(range(3, 3 + self.num_noise_tokens))

    @property
    def open_tokens(self) -> tuple[int, ...]:
        offset = 3 + self.num_noise_tokens
        return tuple(offset + 2 * i for i in range(len(self.bracket_types)))

    @property
    def close_tokens(self) -> tuple[int, ...]:
        return tuple(tok + 1 for tok in self.open_tokens)

    @property
    def vocab_size(self) -> int:
        return 3 + self.num_noise_tokens + 2 * len(self.bracket_types)

    @property
    def max_sequence_length(self) -> int:
        return self.payload_length_for(self.max_length) + 2

    def payload_length_for(self, bracket_length: int) -> int:
        if self.generation_prob >= 1.0:
            return int(bracket_length)
        return int(math.ceil(int(bracket_length) / self.generation_prob))

    @property
    def valid_lengths(self) -> tuple[int, ...]:
        start = self.min_length + (self.min_length % 2)
        return tuple(range(start, self.max_length + 1, 2))

    @property
    def type_names(self) -> tuple[str, ...]:
        return tuple(
            BRACKET_NAME_BY_PAIR.get(pair, f"type_{idx}")
            for idx, pair in enumerate(self.bracket_types)
        )

    def __post_init__(self) -> None:
        self.bracket_types = tuple(self.bracket_types)
        if len(self.bracket_types) != 2:
            raise ValueError("Dyck-(2,3) requires exactly two bracket types")
        if any(len(pair) != 2 for pair in self.bracket_types):
            raise ValueError("Each bracket type must be a two-character pair")
        if len(set(self.bracket_types)) != len(self.bracket_types):
            raise ValueError("bracket_types must be unique")
        if self.min_length < 0:
            raise ValueError("min_length must be non-negative")
        if self.max_length < self.min_length:
            raise ValueError("max_length must be >= min_length")
        if self.max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if not 0.0 < self.generation_prob <= 1.0:
            raise ValueError("generation_prob must be in (0, 1]")
        if self.num_noise_tokens < 0:
            raise ValueError("num_noise_tokens must be non-negative")
        if self.generation_prob < 1.0 and self.num_noise_tokens == 0:
            raise ValueError("num_noise_tokens must be positive when generation_prob < 1")
        if not self.valid_lengths:
            raise ValueError("length range must contain at least one even length")
