def test_package_imports():
    import hse

    assert hse.__all__ == ["analysis", "models", "tasks", "utils"]


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
