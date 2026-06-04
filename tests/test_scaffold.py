def test_package_imports():
    import hse

    assert hse.__all__ == ["analysis", "experiments", "models", "tasks", "utils"]


def test_dyck_sampler_and_rnn_smoke():
    from hse.models import build_model
    from hse.tasks.dyck import DyckConfig, DyckSampler, build_prefix_labels

    cfg = DyckConfig(total_length=8, seq_len=12, dyck_pairs=4, repeat_prob=0.5)
    sampler = DyckSampler(cfg, seed=0)
    batch = sampler.sample(3)
    labels = build_prefix_labels(batch, cfg)
    model = build_model("rnn", vocab_size=sampler.vocab_size, emb_dim=8, hidden_dim=8, layers=1)
    logits = model(batch.tokens)
    states = model.extract_states(batch.tokens)

    assert batch.tokens.shape == (3, 12)
    assert batch.dyck_mask.sum(dim=1).tolist() == [8, 8, 8]
    assert {"left", "right", "height"}.issubset(labels.columns)
    assert logits.shape == (3, 12, sampler.vocab_size)
    assert states.shape == (3, 12, 8)


def test_dyck23_cfg_sampler_generates_valid_strings():
    from hse.tasks.dyck23 import Dyck23Config, Dyck23Sampler, build_prefix_labels, validate_dyck23_tokens

    cfg = Dyck23Config(min_length=0, max_length=12, max_depth=3)
    sampler = Dyck23Sampler(cfg, seed=0)
    batch = sampler.sample(16)
    labels = build_prefix_labels(batch, cfg)

    assert batch.tokens.shape == (16, 14)
    assert batch.target_mask[:, 0].sum().item() == 0
    assert {"depth", "top_type_class", "legal_next_class"}.issubset(labels.columns)
    for row in range(batch.tokens.shape[0]):
        length = int(batch.bracket_lengths[row].item())
        bracket_tokens = batch.tokens[row, 1 : 1 + length].tolist()
        assert validate_dyck23_tokens(bracket_tokens, cfg)
        assert int(batch.dyck_mask[row].sum().item()) == length
