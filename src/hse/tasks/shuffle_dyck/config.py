from __future__ import annotations

from dataclasses import dataclass


BRACKET_NAME_BY_PAIR = {
    "()": "round",
    "[]": "square",
    "{}": "curly",
    "<>": "angle",
}


@dataclass
class ShuffleDyckConfig:
    bracket_types: tuple[str, ...] = ("()", "[]", "{}")
    pairs_per_type: int = 8
    total_length: int = 48
    seq_len: int = 48
    generation_prob: float = 1.0
    n_tasks: int = 512
    prefix_probe_max_len: int = 7
    num_noise_tokens: int = 10
    device: str = "cpu"

    @property
    def num_bracket_types(self) -> int:
        return len(self.bracket_types)

    @property
    def steps_per_type(self) -> int:
        return self.pairs_per_type * 2

    @property
    def total_pairs(self) -> int:
        return self.pairs_per_type * self.num_bracket_types

    @property
    def repeat_prob(self) -> float:
        return self.generation_prob

    @property
    def open_tokens(self) -> tuple[int, ...]:
        return tuple(self.num_noise_tokens + 2 * i for i in range(self.num_bracket_types))

    @property
    def close_tokens(self) -> tuple[int, ...]:
        return tuple(tok + 1 for tok in self.open_tokens)

    @property
    def vocab_size(self) -> int:
        return self.num_noise_tokens + 2 * self.num_bracket_types

    @property
    def type_names(self) -> tuple[str, ...]:
        return tuple(
            BRACKET_NAME_BY_PAIR.get(pair, f"type_{i}")
            for i, pair in enumerate(self.bracket_types)
        )

    def __post_init__(self) -> None:
        self.bracket_types = tuple(self.bracket_types)
        if not self.bracket_types:
            raise ValueError("bracket_types must contain at least one bracket pair")
        if any(len(pair) != 2 for pair in self.bracket_types):
            raise ValueError("Each entry of bracket_types must be a two-character bracket pair")
        if len(set(self.bracket_types)) != len(self.bracket_types):
            raise ValueError("bracket_types must be unique")
        if self.total_length > self.seq_len:
            raise ValueError("total_length cannot exceed seq_len")
        if self.total_length % 2 != 0:
            raise ValueError("total_length must be even for balanced bracket streams")
        if self.total_length != self.steps_per_type * self.num_bracket_types:
            raise ValueError("total_length must equal 2 * pairs_per_type * len(bracket_types)")
        if not (0.0 < float(self.generation_prob) <= 1.0):
            raise ValueError("generation_prob must lie in (0, 1]")
