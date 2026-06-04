from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba as MambaSSM
except ImportError:  # pragma: no cover
    MambaSSM = None


class RecurrentLM(nn.Module):
    def __init__(
        self,
        *,
        model_name: str,
        vocab_size: int,
        emb_dim: int = 128,
        hidden_dim: int = 128,
        layers: int = 3,
        dropout: float = 0.0,
        learned_initial_state: bool = False,
        fused: bool = False,
        **_: object,
    ):
        super().__init__()
        if model_name not in {"rnn", "lstm"}:
            raise ValueError(f"Unsupported recurrent model: {model_name}")
        self.model_name = model_name
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = layers
        self.dropout_p = float(dropout)
        self.fused = bool(fused)
        self.embed = nn.Embedding(vocab_size, emb_dim)
        if self.fused:
            if model_name == "lstm":
                self.rnn = nn.LSTM(
                    input_size=emb_dim,
                    hidden_size=hidden_dim,
                    num_layers=layers,
                    dropout=self.dropout_p if layers > 1 else 0.0,
                    batch_first=True,
                )
            else:
                self.rnn = nn.RNN(
                    input_size=emb_dim,
                    hidden_size=hidden_dim,
                    num_layers=layers,
                    nonlinearity="tanh",
                    dropout=self.dropout_p if layers > 1 else 0.0,
                    batch_first=True,
                )
            self.layers = nn.ModuleList()
        else:
            cell_cls = nn.LSTMCell if model_name == "lstm" else nn.RNNCell
            self.layers = nn.ModuleList(
                [
                    cell_cls(emb_dim if i == 0 else hidden_dim, hidden_dim)
                    for i in range(layers)
                ]
            )
        self.initial_h = (
            nn.Parameter(torch.zeros(layers, hidden_dim))
            if learned_initial_state
            else None
        )
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, *, return_traces: bool = False):
        batch_size, seq_len = x.shape
        emb = F.dropout(self.embed(x), p=self.dropout_p, training=self.training)
        if self.fused:
            if self.initial_h is None:
                h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
            else:
                h0 = self.initial_h.to(device=x.device).unsqueeze(1).expand(-1, batch_size, -1).contiguous()
            if self.model_name == "lstm":
                c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
                h, _ = self.rnn(emb, (h0, c0))
            else:
                h, _ = self.rnn(emb, h0)
            logits = self.output(F.dropout(h, p=self.dropout_p, training=self.training))
            if not return_traces:
                return logits
            return logits, {"h": h.unsqueeze(0)}

        if self.initial_h is None:
            h_states = [torch.zeros(batch_size, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
        else:
            h0 = self.initial_h.to(device=x.device).unsqueeze(1).expand(-1, batch_size, -1)
            h_states = [h0[i].contiguous() for i in range(self.num_layers)]
        c_states = [torch.zeros(batch_size, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
        layer_h = [[] for _ in range(self.num_layers)] if return_traces else None
        layer_c = [[] for _ in range(self.num_layers)] if return_traces and self.model_name == "lstm" else None
        logits = []

        for t in range(seq_len):
            inp = emb[:, t]
            for i, cell in enumerate(self.layers):
                if self.model_name == "lstm":
                    h_new, c_new = cell(inp, (h_states[i], c_states[i]))
                    h_states[i], c_states[i] = h_new, c_new
                    inp = F.dropout(h_new, p=self.dropout_p, training=self.training)
                    if return_traces:
                        layer_h[i].append(h_new)
                        layer_c[i].append(c_new)
                else:
                    h_new = cell(inp, h_states[i])
                    h_states[i] = h_new
                    inp = F.dropout(h_new, p=self.dropout_p, training=self.training)
                    if return_traces:
                        layer_h[i].append(h_new)
            logits.append(self.output(inp))

        logits_t = torch.stack(logits, dim=1)
        if not return_traces:
            return logits_t
        traces = {"h": torch.stack([torch.stack(v, dim=1) for v in layer_h], dim=0)}
        if layer_c is not None:
            traces["c"] = torch.stack([torch.stack(v, dim=1) for v in layer_c], dim=0)
        return logits_t, traces

    @torch.no_grad()
    def extract_states(self, x: torch.Tensor, *, layer_index: int = -1, state_kind: str = "h") -> torch.Tensor:
        if self.fused:
            if state_kind != "h":
                raise ValueError("Fused recurrent path only exposes state_kind='h'")
            if layer_index not in {-1, self.num_layers - 1}:
                raise ValueError("Fused recurrent path only exposes the final layer sequence")
            _, traces = self.forward(x, return_traces=True)
            return traces["h"][0]
        _, traces = self.forward(x, return_traces=True)
        if state_kind not in traces:
            raise ValueError(f"State kind {state_kind!r} unavailable; options={sorted(traces)}")
        layer_index = self.num_layers + layer_index if layer_index < 0 else layer_index
        return traces[state_kind][layer_index]


class TransformerLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        emb_dim: int = 128,
        hidden_dim: int = 128,
        layers: int = 3,
        n_heads: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.0,
        max_positions: int = 4096,
        pos_encoding: str = "learned",
        embed_scale: bool = False,
        final_layer_norm: bool = False,
        activation: str = "gelu",
        **_: object,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = layers
        self.embed_scale = bool(embed_scale)
        self.pos_encoding_type = pos_encoding
        self.embed = nn.Embedding(vocab_size, emb_dim)
        if pos_encoding == "sinusoidal":
            self.register_buffer(
                "sinusoidal_pos",
                _sinusoidal_positions(max_positions, emb_dim),
                persistent=False,
            )
            self.pos_embed = None
        elif pos_encoding == "learned":
            self.pos_embed = nn.Embedding(max_positions, emb_dim)
            self.sinusoidal_pos = None
        else:
            raise ValueError("pos_encoding must be 'learned' or 'sinusoidal'")
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.ModuleList([encoder_layer if i == 0 else _clone_encoder_layer(encoder_layer) for i in range(layers)])
        self.final_norm = nn.LayerNorm(emb_dim) if final_layer_norm else nn.Identity()
        self.output = nn.Linear(emb_dim, vocab_size)

    def forward(self, x: torch.Tensor, *, return_traces: bool = False):
        batch_size, seq_len = x.shape
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        h = self.embed(x)
        if self.embed_scale:
            h = h * math.sqrt(self.emb_dim)
        if self.pos_encoding_type == "sinusoidal":
            h = h + self.sinusoidal_pos[:seq_len].to(device=x.device).unsqueeze(0)
        else:
            h = h + self.pos_embed(pos)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        traces = []
        for layer in self.layers:
            h = layer(h, src_mask=mask)
            if return_traces:
                traces.append(h)
        h = self.final_norm(h)
        if return_traces and traces:
            traces[-1] = h
        logits = self.output(h)
        if not return_traces:
            return logits
        return logits, {"h": torch.stack(traces, dim=0)}

    @torch.no_grad()
    def extract_states(self, x: torch.Tensor, *, layer_index: int = -1, state_kind: str = "h") -> torch.Tensor:
        if state_kind != "h":
            raise ValueError("TransformerLM only exposes state_kind='h'")
        _, traces = self.forward(x, return_traces=True)
        layer_index = self.num_layers + layer_index if layer_index < 0 else layer_index
        return traces["h"][layer_index]


class MambaLikeLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        emb_dim: int = 128,
        hidden_dim: int = 128,
        layers: int = 3,
        state_dim: int = 16,
        expansion_factor: int = 2,
        **_: object,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = layers
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.input_proj = nn.Linear(emb_dim, hidden_dim) if emb_dim != hidden_dim else nn.Identity()
        if MambaSSM is not None:
            self.layers = nn.ModuleList(
                [MambaSSM(d_model=hidden_dim, d_state=state_dim, expand=expansion_factor) for _ in range(layers)]
            )
            self.uses_mamba_ssm = True
        else:
            self.layers = nn.ModuleList([SelectiveStateBlock(hidden_dim, expansion_factor) for _ in range(layers)])
            self.uses_mamba_ssm = False
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, *, return_traces: bool = False):
        h = self.input_proj(self.embed(x))
        traces = []
        for layer in self.layers:
            h = layer(h)
            if return_traces:
                traces.append(h)
        logits = self.output(h)
        if not return_traces:
            return logits
        return logits, {"h": torch.stack(traces, dim=0)}

    @torch.no_grad()
    def extract_states(self, x: torch.Tensor, *, layer_index: int = -1, state_kind: str = "h") -> torch.Tensor:
        if state_kind != "h":
            raise ValueError("MambaLikeLM only exposes state_kind='h'")
        _, traces = self.forward(x, return_traces=True)
        layer_index = self.num_layers + layer_index if layer_index < 0 else layer_index
        return traces["h"][layer_index]


class OfficialMambaUnavailableError(ImportError):
    """Raised when the official mamba-ssm package is required but unavailable."""


class SelectiveStateBlock(nn.Module):
    """Small gated causal state block used when mamba-ssm is unavailable."""

    def __init__(self, hidden_dim: int, expansion_factor: int = 2):
        super().__init__()
        inner = hidden_dim * expansion_factor
        self.in_proj = nn.Linear(hidden_dim, inner * 2)
        self.state_proj = nn.Linear(inner, inner)
        self.out_proj = nn.Linear(inner, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        u, gate = self.in_proj(x).chunk(2, dim=-1)
        u = torch.nn.functional.silu(u)
        gate = torch.sigmoid(gate)
        state = torch.zeros_like(u[:, 0])
        outputs = []
        for t in range(u.shape[1]):
            candidate = torch.tanh(self.state_proj(state) + u[:, t])
            state = gate[:, t] * candidate + (1.0 - gate[:, t]) * state
            outputs.append(state)
        y = self.out_proj(torch.stack(outputs, dim=1))
        return residual + y


def build_model(model_name: str, vocab_size: int, **kwargs: object) -> nn.Module:
    model_name = model_name.lower()
    if model_name in {"rnn", "lstm"}:
        return RecurrentLM(model_name=model_name, vocab_size=vocab_size, **kwargs)
    if model_name == "transformer":
        return TransformerLM(vocab_size=vocab_size, **kwargs)
    if model_name == "mamba":
        require_official = bool(kwargs.pop("require_official_mamba", False))
        if require_official and MambaSSM is None:
            raise OfficialMambaUnavailableError(
                "Official Mamba requires the 'mamba-ssm' package, which is not installed. "
                "The upstream package currently targets Linux + NVIDIA GPU + CUDA. "
                "Use a Linux/WSL CUDA environment, then install with: "
                "pip install causal-conv1d>=1.4.0 mamba-ssm --no-build-isolation"
            )
        return MambaLikeLM(vocab_size=vocab_size, **kwargs)
    if model_name == "mamba_like":
        return MambaLikeLM(vocab_size=vocab_size, **kwargs)
    raise ValueError(f"Unknown model_name: {model_name}")


def _clone_encoder_layer(layer: nn.TransformerEncoderLayer) -> nn.TransformerEncoderLayer:
    import copy

    return copy.deepcopy(layer)


def _sinusoidal_positions(max_positions: int, dim: int) -> torch.Tensor:
    position = torch.arange(max_positions, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(max_positions, dim, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe
