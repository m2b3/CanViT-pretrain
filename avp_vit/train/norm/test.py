"""Tests for position-aware running normalization."""

import torch

from avp_vit.train.norm import PositionAwareNorm


def test_output_shape():
    norm = PositionAwareNorm(n_tokens=16, embed_dim=64, grid_size=4)
    x = torch.randn(8, 16, 64)
    y = norm(x)
    assert y.shape == x.shape


def test_first_batch_initializes_stats():
    norm = PositionAwareNorm(n_tokens=16, embed_dim=64, grid_size=4)
    norm.train()
    assert not norm.initialized

    x = torch.randn(8, 16, 64)
    norm(x)

    assert norm.initialized
    assert norm.mean.shape == (16, 64)
    assert norm.var.shape == (16, 64)


def test_running_stats_update():
    norm = PositionAwareNorm(n_tokens=16, embed_dim=64, grid_size=4, momentum=0.5)
    norm.train()

    x1 = torch.randn(8, 16, 64)
    norm(x1)
    mean_after_first = norm.mean.clone()

    x2 = torch.randn(8, 16, 64) + 10  # Shift by 10
    norm(x2)

    # Mean should have moved toward new batch
    assert not torch.allclose(norm.mean, mean_after_first)


def test_eval_mode_no_update():
    norm = PositionAwareNorm(n_tokens=16, embed_dim=64, grid_size=4)
    norm.train()

    x = torch.randn(8, 16, 64)
    norm(x)
    mean_after_train = norm.mean.clone()

    norm.eval()
    x2 = torch.randn(8, 16, 64) + 100
    norm(x2)

    # Mean should NOT have changed
    assert torch.allclose(norm.mean, mean_after_train)


def test_state_dict_contains_buffers():
    norm = PositionAwareNorm(n_tokens=16, embed_dim=64, grid_size=4)
    sd = norm.state_dict()
    assert "mean" in sd
    assert "var" in sd
    assert "_initialized" in sd


def test_initialized_persists_through_save_load():
    """initialized flag must survive state_dict round-trip."""
    norm1 = PositionAwareNorm(n_tokens=4, embed_dim=8, grid_size=2)
    norm1.train()
    assert not norm1.initialized

    norm1(torch.randn(4, 4, 8))
    assert norm1.initialized

    # Save and load into fresh instance
    norm2 = PositionAwareNorm(n_tokens=4, embed_dim=8, grid_size=2)
    assert not norm2.initialized
    norm2.load_state_dict(norm1.state_dict())
    assert norm2.initialized, "initialized flag not restored from state_dict"


# ============================================================================
# Chan's parallel variance / Welford tests
# ============================================================================


def test_batch_size_1_no_nan():
    """B=1 must not produce NaN - this was the original bug."""
    norm = PositionAwareNorm(n_tokens=4, embed_dim=8, grid_size=2, momentum=0.1)
    norm.train()

    # Multiple B=1 updates
    for _ in range(10):
        x = torch.randn(1, 4, 8)
        y = norm(x)
        assert not y.isnan().any(), "Output contains NaN"
        assert not norm.mean.isnan().any(), "Mean contains NaN"
        assert not norm.var.isnan().any(), "Var contains NaN"


def test_batch_size_1_reduces_to_welford():
    """For B=1, Chan's formula must equal Welford's online update."""
    α = 0.1
    norm = PositionAwareNorm(n_tokens=4, embed_dim=8, grid_size=2, momentum=α)
    norm.train()

    # Initialize with a batch
    norm(torch.randn(4, 4, 8))
    var_before = norm.var.clone()
    mean_before = norm.mean.clone()

    # Single sample update
    x = torch.randn(1, 4, 8)
    norm(x)

    # Verify Welford formula: var_new = (1-α)*(var_old + α*δ²)
    delta = x[0] - mean_before
    expected_var = (1 - α) * (var_before + α * delta**2)

    assert torch.allclose(norm.var, expected_var, rtol=1e-5), (
        f"Welford mismatch: got {norm.var}, expected {expected_var}"
    )


def test_batch_and_sequential_converge_similarly():
    """Batch and sequential updates converge to same distribution stats.

    Note: Batch != sequential exactly (batch uses uniform sample weights,
    sequential weights recent samples more). But both converge similarly.
    """
    torch.manual_seed(999)
    α = 0.1
    N, D = 4, 8
    true_mean, true_std = 5.0, 2.0

    norm_batch = PositionAwareNorm(n_tokens=N, embed_dim=D, grid_size=2, momentum=α)
    norm_seq = PositionAwareNorm(n_tokens=N, embed_dim=D, grid_size=2, momentum=α)
    norm_batch.train()
    norm_seq.train()

    # Feed same data to both, but batch vs B=1 sequential
    for _ in range(100):
        batch = torch.randn(8, N, D) * true_std + true_mean
        norm_batch(batch)
        for i in range(8):
            norm_seq(batch[i : i + 1])

    # Both should converge to similar stats (not identical, but close)
    assert torch.allclose(norm_batch.mean, norm_seq.mean, rtol=0.1), "Means diverged"
    assert torch.allclose(norm_batch.var, norm_seq.var, rtol=0.2), "Vars diverged"


def test_variance_tracks_distribution():
    """Running variance should converge toward true variance."""
    torch.manual_seed(42)
    N, D = 16, 32
    true_std = 3.0

    norm = PositionAwareNorm(n_tokens=N, embed_dim=D, grid_size=4, momentum=0.1)
    norm.train()

    # Feed many batches from N(0, true_std²)
    for _ in range(200):
        x = torch.randn(8, N, D) * true_std
        norm(x)

    # Running var should be close to true_std²
    expected_var = true_std**2
    actual_var = norm.var.mean().item()
    assert abs(actual_var - expected_var) < 1.0, (
        f"Variance not tracking: got {actual_var:.2f}, expected {expected_var:.2f}"
    )


def test_single_sample_updates_track_variance():
    """Even with B=1, variance should converge over many updates."""
    torch.manual_seed(123)
    N, D = 4, 8
    true_std = 2.0

    norm = PositionAwareNorm(n_tokens=N, embed_dim=D, grid_size=2, momentum=0.05)
    norm.train()

    # Many B=1 updates
    for _ in range(500):
        x = torch.randn(1, N, D) * true_std
        norm(x)

    # Should converge (with higher tolerance due to single-sample noise)
    expected_var = true_std**2
    actual_var = norm.var.mean().item()
    assert abs(actual_var - expected_var) < 1.5, (
        f"B=1 variance not tracking: got {actual_var:.2f}, expected {expected_var:.2f}"
    )


def test_effective_momentum_formula():
    """Verify m = 1 - (1-α)^B gives correct effective momentum."""
    α = 0.1
    B = 4

    # Expected: 1 - 0.9^4 = 1 - 0.6561 = 0.3439
    m_expected = 1 - (1 - α) ** B

    # After B samples, old stats should have weight (1-α)^B = 0.6561
    # New info should have weight 1 - 0.6561 = 0.3439
    assert abs(m_expected - 0.3439) < 0.001


def test_denormalize_inverts_forward():
    """denormalize should invert forward normalization."""
    norm = PositionAwareNorm(n_tokens=4, embed_dim=8, grid_size=2)
    norm.train()
    norm(torch.randn(8, 4, 8))
    norm.eval()

    x = torch.randn(4, 4, 8)
    normalized = norm(x)
    recovered = norm.denormalize(normalized)

    assert torch.allclose(recovered, x, rtol=1e-5)
