from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
import copy
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from hse.experiments.dyck23 import count_trainable_parameters, run_dyck23_probes
from hse.tasks.dyck23 import Dyck23Batch, Dyck23Config, Dyck23Sampler, build_prefix_labels
from hse.utils.io import save_json
from hse.utils.labels_io import save_labels


DYCK2_TRANSFORMER_MODEL_SPEC = {
    "layers": 5,
    "emb_dim": 32,
    "hidden_dim": 32,
    "n_heads": 8,
    "ffn_dim": 128,
    "dropout": 0.1,
    "pos_encoding": "sinusoidal",
    "embed_scale": True,
    "final_layer_norm": True,
    "state_kind": "h",
}


def run_dyck2_transformer_suite(
    *,
    seed: int = 0,
    length_bins: dict[str, dict],
    train_examples: int = 10_000,
    test_examples: int = 512,
    probe_examples: int = 512,
    steps: int | None = 15_000,
    epochs: int = 20,
    batch_size: int = 64,
    train_micro_batch_size: int = 2,
    eval_batch_size: int = 2,
    learning_rate: float = 3e-4,
    grad_clip: float = 5.0,
    eval_every_epochs: int = 1,
    extract_batch_size: int = 2,
    amp_dtype: str | None = "bfloat16",
    gradient_checkpointing: bool = True,
    max_probe_rows: int = 20_000,
    device: str | None = None,
    results_root: str | Path | None = None,
    setting_name: str = "dyck2_cfg_transformer_next_token",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train the Dyck-2 CFG Transformer with long-sequence memory controls.

    This is intentionally separate from the Dyck-(2,3) suite so the original
    experiment defaults and model path remain untouched.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    results_root = Path(results_root or Path("results") / setting_name)
    results_root.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict] = []
    probe_rows: list[dict] = []
    in_progress_runs_path = results_root / "runs.in_progress.csv"
    in_progress_probe_path = results_root / "probe_summary.in_progress.csv"

    for bin_idx, (bin_name, bin_kwargs) in enumerate(length_bins.items()):
        cfg = Dyck23Config(**bin_kwargs, device="cpu")
        train_batch = Dyck23Sampler(cfg, seed=seed + 10_000 * bin_idx).sample(train_examples)
        test_batch = Dyck23Sampler(cfg, seed=seed + 10_000 * bin_idx + 1).sample(test_examples)
        probe_batch = Dyck23Sampler(cfg, seed=seed + 10_000 * bin_idx + 2).sample(probe_examples)

        bin_dir = results_root / bin_name
        bin_dir.mkdir(parents=True, exist_ok=True)
        save_json(
            {
                "bin": bin_name,
                "task": asdict(cfg),
                "train_examples": train_examples,
                "test_examples": test_examples,
                "probe_examples": probe_examples,
                "valid_lengths": list(cfg.valid_lengths),
            },
            bin_dir / "data_summary.json",
        )

        spec = dict(DYCK2_TRANSFORMER_MODEL_SPEC)
        model_kwargs = {k: v for k, v in spec.items() if k != "state_kind"}
        torch.manual_seed(seed + 1_000 * bin_idx)
        model = Dyck2MemoryEfficientTransformerLM(
            vocab_size=cfg.vocab_size,
            gradient_checkpointing=gradient_checkpointing,
            **model_kwargs,
        )
        run_dir = bin_dir / f"transformer_seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        total_params = count_trainable_parameters(model)
        backbone_params = count_trainable_parameters(model, exclude_output=True)
        save_json(
            {
                "setting_name": setting_name,
                "length_bin": bin_name,
                "task": asdict(cfg),
                "model_name": "transformer",
                "model": spec,
                "seed": seed,
                "training_steps": steps,
                "epochs": epochs,
                "batch_size": batch_size,
                "train_micro_batch_size": train_micro_batch_size,
                "eval_batch_size": eval_batch_size,
                "effective_batch_size": batch_size,
                "learning_rate": learning_rate,
                "grad_clip": grad_clip,
                "amp_dtype": amp_dtype,
                "gradient_checkpointing": gradient_checkpointing,
                "device": device,
                "train_examples": train_examples,
                "test_examples": test_examples,
                "probe_examples": probe_examples,
                "trainable_parameters": total_params,
                "backbone_parameters_excluding_lm_head": backbone_params,
            },
            run_dir / "config.json",
        )

        print(f"\n[{bin_name}] training Dyck-2 transformer ({backbone_params:,} backbone params)")
        started = time.time()
        train_log, eval_metrics = train_dyck2_transformer_model(
            model=model,
            train_batch=train_batch,
            test_batch=test_batch,
            vocab_size=cfg.vocab_size,
            steps=steps,
            epochs=epochs,
            batch_size=batch_size,
            train_micro_batch_size=train_micro_batch_size,
            eval_batch_size=eval_batch_size,
            lr=learning_rate,
            grad_clip=grad_clip,
            eval_every_epochs=eval_every_epochs,
            amp_dtype=amp_dtype,
            device=device,
            run_dir=run_dir,
        )

        hidden, labels = extract_dyck2_hidden_states(
            model=model,
            batch=probe_batch,
            config=cfg,
            state_kind=spec["state_kind"],
            batch_size=extract_batch_size,
            amp_dtype=amp_dtype,
            device=device,
            run_dir=run_dir,
        )
        probe_summary, compression_summary = run_dyck23_probes(
            hidden,
            labels,
            seed=seed,
            max_rows=max_probe_rows,
            run_dir=run_dir,
        )

        elapsed = time.time() - started
        save_json(
            {
                "train": train_log,
                "eval": eval_metrics,
                "probe": probe_summary,
                "compression": compression_summary,
                "elapsed_seconds": elapsed,
            },
            run_dir / "metrics.json",
        )

        run_row = {
            "length_bin": bin_name,
            "model": "transformer",
            "seed": seed,
            "run_dir": str(run_dir.resolve()),
            "loss": eval_metrics["loss"],
            "accuracy": eval_metrics["accuracy"],
            "dyck_accuracy": eval_metrics["dyck_accuracy"],
            "hidden_rows": int(hidden.shape[0]),
            "hidden_dim": int(hidden.shape[1]),
            "params": total_params,
            "backbone_params": backbone_params,
            "elapsed_seconds": elapsed,
        }
        probe_row = {"length_bin": bin_name, "model": "transformer", "seed": seed, **probe_summary, **compression_summary}
        run_rows.append(run_row)
        probe_rows.append(probe_row)
        pd.DataFrame(run_rows).to_csv(in_progress_runs_path, index=False)
        pd.DataFrame(probe_rows).to_csv(in_progress_probe_path, index=False)
        print(f"[{bin_name}] finished transformer: acc={eval_metrics['accuracy']:.3f}, loss={eval_metrics['loss']:.3f}")

    runs_df = pd.DataFrame(run_rows)
    probe_df = pd.DataFrame(probe_rows)
    runs_df.to_csv(results_root / "runs.csv", index=False)
    probe_df.to_csv(results_root / "probe_summary.csv", index=False)
    return runs_df, probe_df


def train_dyck2_transformer_model(
    *,
    model: nn.Module,
    train_batch: Dyck23Batch,
    test_batch: Dyck23Batch,
    vocab_size: int,
    steps: int | None,
    epochs: int,
    batch_size: int,
    train_micro_batch_size: int,
    eval_batch_size: int,
    lr: float,
    grad_clip: float,
    eval_every_epochs: int,
    amp_dtype: str | None,
    device: str,
    run_dir: Path,
) -> tuple[list[dict], dict]:
    device_t = torch.device(device)
    model.to(device_t)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(0)
    resolved_amp_dtype = _resolve_amp_dtype(amp_dtype, device_t)
    scaler = _make_grad_scaler(enabled=resolved_amp_dtype is torch.float16 and device_t.type == "cuda")
    train_size = int(train_batch.tokens.shape[0])
    log: list[dict] = []
    best_loss = float("inf")
    best_state = None
    best_step = 0
    best_epoch = 0.0
    global_step = 0

    if steps is not None:
        eval_every_steps = max(1, min(1000, int(steps)))
        order = torch.randperm(train_size, generator=generator)
        cursor = 0
        train_loss_sum = 0.0
        train_token_count = 0
        for step in range(1, int(steps) + 1):
            model.train()
            if cursor + batch_size > train_size:
                order = torch.randperm(train_size, generator=generator)
                cursor = 0
            idx = order[cursor : cursor + batch_size]
            cursor += batch_size
            loss, token_count = _train_effective_batch(
                model=model,
                train_batch=train_batch,
                idx=idx,
                vocab_size=vocab_size,
                optimizer=optimizer,
                criterion=criterion,
                grad_clip=grad_clip,
                micro_batch_size=train_micro_batch_size,
                amp_dtype=resolved_amp_dtype,
                scaler=scaler,
                device=device_t,
            )
            train_loss_sum += loss * token_count
            train_token_count += token_count
            global_step = step

            should_eval = step == 1 or step % eval_every_steps == 0 or step == int(steps)
            if should_eval:
                eval_metrics = evaluate_dyck2_transformer_model(
                    model=model,
                    batch=test_batch,
                    vocab_size=vocab_size,
                    batch_size=eval_batch_size,
                    amp_dtype=amp_dtype,
                    device=device,
                )
                train_loss = train_loss_sum / max(train_token_count, 1)
                row = {"epoch": float(step * batch_size / train_size), "step": step, "train_loss": train_loss, **eval_metrics}
                log.append(row)
                print(
                    f"  step {step:05d}/{int(steps)}: train_loss={train_loss:.4f} "
                    f"test_loss={eval_metrics['loss']:.4f} acc={eval_metrics['accuracy']:.3f}"
                )
                train_loss_sum = 0.0
                train_token_count = 0
                if eval_metrics["loss"] < best_loss:
                    best_loss = eval_metrics["loss"]
                    best_step = step
                    best_epoch = row["epoch"]
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    else:
        for epoch in range(1, epochs + 1):
            model.train()
            permutation = torch.randperm(train_size, generator=generator)
            train_loss_sum = 0.0
            train_token_count = 0
            for start in range(0, train_size, batch_size):
                idx = permutation[start : start + batch_size]
                loss, token_count = _train_effective_batch(
                    model=model,
                    train_batch=train_batch,
                    idx=idx,
                    vocab_size=vocab_size,
                    optimizer=optimizer,
                    criterion=criterion,
                    grad_clip=grad_clip,
                    micro_batch_size=train_micro_batch_size,
                    amp_dtype=resolved_amp_dtype,
                    scaler=scaler,
                    device=device_t,
                )
                train_loss_sum += loss * token_count
                train_token_count += token_count
                global_step += 1

            should_eval = epoch == 1 or epoch % eval_every_epochs == 0 or epoch == epochs
            if should_eval:
                eval_metrics = evaluate_dyck2_transformer_model(
                    model=model,
                    batch=test_batch,
                    vocab_size=vocab_size,
                    batch_size=eval_batch_size,
                    amp_dtype=amp_dtype,
                    device=device,
                )
                train_loss = train_loss_sum / max(train_token_count, 1)
                row = {"epoch": epoch, "step": global_step, "train_loss": train_loss, **eval_metrics}
                log.append(row)
                print(
                    f"  epoch {epoch:03d}/{epochs}: train_loss={train_loss:.4f} "
                    f"test_loss={eval_metrics['loss']:.4f} acc={eval_metrics['accuracy']:.3f}"
                )
                if eval_metrics["loss"] < best_loss:
                    best_loss = eval_metrics["loss"]
                    best_step = global_step
                    best_epoch = float(epoch)
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "epoch": epochs, "step": global_step}, ckpt_dir / "model_final.pt")
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save({"model": best_state, "step": best_step, "epoch": best_epoch, "loss": best_loss}, ckpt_dir / "model_best.pt")
    eval_metrics = evaluate_dyck2_transformer_model(
        model=model,
        batch=test_batch,
        vocab_size=vocab_size,
        batch_size=eval_batch_size,
        amp_dtype=amp_dtype,
        device=device,
    )
    return log, eval_metrics


class Dyck2MemoryEfficientTransformerLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        emb_dim: int = 32,
        hidden_dim: int = 32,
        layers: int = 5,
        n_heads: int = 8,
        ffn_dim: int = 128,
        dropout: float = 0.1,
        max_positions: int = 4096,
        pos_encoding: str = "sinusoidal",
        embed_scale: bool = True,
        final_layer_norm: bool = True,
        activation: str = "gelu",
        gradient_checkpointing: bool = True,
        **_: object,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = layers
        self.embed_scale = bool(embed_scale)
        self.pos_encoding_type = pos_encoding
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.embed = nn.Embedding(vocab_size, emb_dim)
        if pos_encoding == "sinusoidal":
            self.register_buffer("sinusoidal_pos", _sinusoidal_positions(max_positions, emb_dim), persistent=False)
            self.pos_embed = None
        elif pos_encoding == "learned":
            self.pos_embed = nn.Embedding(max_positions, emb_dim)
            self.sinusoidal_pos = None
        else:
            raise ValueError("pos_encoding must be 'learned' or 'sinusoidal'")
        layer = Dyck2CausalTransformerBlock(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation=activation,
        )
        self.layers = nn.ModuleList([layer if i == 0 else copy.deepcopy(layer) for i in range(layers)])
        self.final_norm = nn.LayerNorm(emb_dim) if final_layer_norm else nn.Identity()
        self.output = nn.Linear(emb_dim, vocab_size)

    def forward(self, x: torch.Tensor, *, return_traces: bool = False):
        h = self._embed_tokens(x)
        traces = []
        for layer in self.layers:
            if self.training and self.gradient_checkpointing and not return_traces:
                h = checkpoint(lambda y, block=layer: block(y), h, use_reentrant=False)
            else:
                h = layer(h)
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
            raise ValueError("Dyck2MemoryEfficientTransformerLM only exposes state_kind='h'")
        layer_index = self.num_layers + layer_index if layer_index < 0 else layer_index
        if layer_index == self.num_layers - 1:
            h = self._embed_tokens(x)
            for layer in self.layers:
                h = layer(h)
            return self.final_norm(h)
        _, traces = self.forward(x, return_traces=True)
        return traces["h"][layer_index]

    def _embed_tokens(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        h = self.embed(x)
        if self.embed_scale:
            h = h * math.sqrt(self.emb_dim)
        if self.pos_encoding_type == "sinusoidal":
            h = h + self.sinusoidal_pos[:seq_len].to(device=x.device).unsqueeze(0)
        else:
            h = h + self.pos_embed(pos)
        return h


class Dyck2CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.in_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_p = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q = self._split_heads(q, batch_size, seq_len)
        k = self._split_heads(k, batch_size, seq_len)
        v = self._split_heads(v, batch_size, seq_len)
        dropout_p = self.dropout_p if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(y)

    def _split_heads(self, x: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        return x.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)


class Dyck2CausalTransformerBlock(nn.Module):
    def __init__(self, *, d_model: int, nhead: int, dim_feedforward: int, dropout: float, activation: str):
        super().__init__()
        self.self_attn = Dyck2CausalSelfAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _activation_fn(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout1(self.self_attn(self.norm1(x)))
        x = x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(self.norm2(x))))))
        return x


def _train_effective_batch(
    *,
    model: nn.Module,
    train_batch: Dyck23Batch,
    idx: torch.Tensor,
    vocab_size: int,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    grad_clip: float,
    micro_batch_size: int,
    amp_dtype: torch.dtype | None,
    scaler,
    device: torch.device,
) -> tuple[float, int]:
    optimizer.zero_grad(set_to_none=True)
    total_token_count = int(train_batch.target_mask[idx, 1:].sum().item())
    if total_token_count == 0:
        return 0.0, 0

    loss_sum = 0.0
    for start in range(0, int(idx.shape[0]), micro_batch_size):
        micro_idx = idx[start : start + micro_batch_size]
        tokens = train_batch.tokens[micro_idx].to(device)
        target_mask = train_batch.target_mask[micro_idx].to(device)
        token_count = int(target_mask[:, 1:].sum().item())
        if token_count == 0:
            continue
        with _autocast_context(device, amp_dtype):
            logits = model(tokens)
            loss = _masked_ce_mean(criterion, logits, tokens, target_mask, vocab_size)
            weighted_loss = loss * (token_count / total_token_count)
        loss_sum += float(loss.detach().item()) * token_count
        if scaler.is_enabled():
            scaler.scale(weighted_loss).backward()
        else:
            weighted_loss.backward()

    if scaler.is_enabled():
        scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    if scaler.is_enabled():
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    return loss_sum / total_token_count, total_token_count


@torch.no_grad()
def evaluate_dyck2_transformer_model(
    *,
    model: nn.Module,
    batch: Dyck23Batch,
    vocab_size: int,
    batch_size: int,
    amp_dtype: str | None,
    device: str,
) -> dict[str, float]:
    device_t = torch.device(device)
    resolved_amp_dtype = _resolve_amp_dtype(amp_dtype, device_t)
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = 0.0
    target_count = 0
    correct = 0
    dyck_correct = 0
    dyck_count = 0

    for start in range(0, int(batch.tokens.shape[0]), batch_size):
        stop = min(start + batch_size, int(batch.tokens.shape[0]))
        tokens = batch.tokens[start:stop].to(device_t)
        target_mask = batch.target_mask[start:stop].to(device_t)
        dyck_mask = batch.dyck_mask[start:stop].to(device_t)
        with _autocast_context(device_t, resolved_amp_dtype):
            logits = model(tokens)
        pred = logits[:, :-1].argmax(dim=-1)
        target = tokens[:, 1:]
        valid = target_mask[:, 1:].bool()
        flat_logits = logits[:, :-1].float().reshape(-1, vocab_size)
        flat_target = target.reshape(-1)
        flat_valid = valid.reshape(-1)
        loss_sum += float(criterion(flat_logits[flat_valid], flat_target[flat_valid]).item())
        target_count += int(flat_valid.sum().item())
        correct += int((pred[valid] == target[valid]).sum().item())

        dyck_valid = dyck_mask[:, 1:].bool()
        dyck_count += int(dyck_valid.sum().item())
        if bool(dyck_valid.any()):
            dyck_correct += int((pred[dyck_valid] == target[dyck_valid]).sum().item())

    return {
        "loss": loss_sum / max(target_count, 1),
        "accuracy": correct / max(target_count, 1),
        "dyck_accuracy": dyck_correct / dyck_count if dyck_count else float("nan"),
        "tokens": float(target_count),
    }


@torch.no_grad()
def extract_dyck2_hidden_states(
    *,
    model: nn.Module,
    batch: Dyck23Batch,
    config: Dyck23Config,
    state_kind: str,
    batch_size: int,
    amp_dtype: str | None,
    device: str,
    run_dir: Path | None = None,
) -> tuple[torch.Tensor, pd.DataFrame]:
    device_t = torch.device(device)
    resolved_amp_dtype = _resolve_amp_dtype(amp_dtype, device_t)
    model.to(device_t)
    model.eval()
    all_hidden = []
    all_labels = []

    for start in range(0, int(batch.tokens.shape[0]), batch_size):
        stop = min(start + batch_size, int(batch.tokens.shape[0]))
        local = _slice_batch(batch, start, stop)
        tokens = local.tokens.to(device_t)
        with _autocast_context(device_t, resolved_amp_dtype):
            states = model.extract_states(tokens, layer_index=-1, state_kind=state_kind).detach().float().cpu()
        labels = build_prefix_labels(local, config=config)
        labels["example_id"] += start
        local_ids = torch.tensor(labels["example_id"].to_numpy() - start, dtype=torch.long)
        positions = torch.tensor(labels["position"].to_numpy(), dtype=torch.long)
        all_hidden.append(states[local_ids, positions].contiguous())
        all_labels.append(labels)

    hidden = torch.cat(all_hidden, dim=0)
    labels_df = pd.concat(all_labels, ignore_index=True)
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(hidden, run_dir / "hidden_states.pt")
        save_labels(labels_df, run_dir / "labels")
    return hidden, labels_df


def _masked_ce_mean(
    criterion: nn.CrossEntropyLoss,
    logits: torch.Tensor,
    tokens: torch.Tensor,
    target_mask: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    flat_logits = logits[:, :-1].float().reshape(-1, vocab_size)
    flat_target = tokens[:, 1:].reshape(-1)
    flat_mask = target_mask[:, 1:].reshape(-1).bool()
    return criterion(flat_logits[flat_mask], flat_target[flat_mask])


def _slice_batch(batch: Dyck23Batch, start: int, stop: int) -> Dyck23Batch:
    return Dyck23Batch(
        tokens=batch.tokens[start:stop],
        dyck_mask=batch.dyck_mask[start:stop],
        target_mask=batch.target_mask[start:stop],
        lengths=batch.lengths[start:stop],
        bracket_lengths=batch.bracket_lengths[start:stop],
        dyck_steps=batch.dyck_steps[start:stop],
        bracket_type_ids=batch.bracket_type_ids[start:stop],
    )


def _resolve_amp_dtype(name: str | None, device: torch.device) -> torch.dtype | None:
    if name is None or str(name).lower() in {"none", "false", "off"}:
        return None
    normalized = str(name).lower()
    if normalized in {"bf16", "bfloat16"}:
        if device.type == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return None
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16 if device.type == "cuda" else None
    raise ValueError(f"Unsupported amp_dtype={name!r}; use None, 'bfloat16', or 'float16'.")


def _autocast_context(device: torch.device, dtype: torch.dtype | None):
    if dtype is None or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def _make_grad_scaler(*, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _activation_fn(name: str):
    normalized = name.lower()
    if normalized == "gelu":
        return F.gelu
    if normalized == "relu":
        return F.relu
    raise ValueError(f"Unsupported activation={name!r}")


def _sinusoidal_positions(max_positions: int, dim: int) -> torch.Tensor:
    position = torch.arange(max_positions, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(max_positions, dim, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe
