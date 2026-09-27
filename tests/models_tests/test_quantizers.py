import torch

from relational_compression.models.image_autoencoder import BinaryImageAutoencoder
from relational_compression.models.quantizers import MeanGroupBinaryQuantizer


def test_per_channel_mean_group_centers_each_sign_at_hard_value():
    logits = torch.tensor(
        [
            [
                [[-3.0, -1.0], [1.0, 5.0]],
                [[-4.0, 2.0], [-2.0, 6.0]],
            ]
        ]
    )
    quantizer = MeanGroupBinaryQuantizer(stochastic=False, grouping="per_channel")
    quantizer.train()
    output = quantizer(logits)

    for channel in range(logits.shape[1]):
        hard = output.hard[:, channel]
        relaxed = output.relaxed[:, channel]
        assert torch.allclose(relaxed[hard > 0].mean(), torch.tensor(1.0))
        assert torch.allclose(relaxed[hard < 0].mean(), torch.tensor(-1.0))


def test_eval_path_is_exactly_binary_and_deterministic():
    logits = torch.tensor([[[[-1.0, 0.2], [3.0, -4.0]]]])
    quantizer = MeanGroupBinaryQuantizer(stochastic=True)
    quantizer.eval()
    first = quantizer(logits)
    second = quantizer(logits)

    assert torch.equal(first.relaxed, first.hard)
    assert torch.equal(first.hard, second.hard)
    assert set(first.hard.unique().tolist()) <= {-1.0, 1.0}


def test_monte_carlo_sign_matches_bernoulli_probability_empirically():
    torch.manual_seed(0)
    logits = torch.full((20000, 1, 1, 1), torch.logit(torch.tensor(0.7)))
    quantizer = MeanGroupBinaryQuantizer(stochastic=True, grouping="per_channel")
    quantizer.train()
    output = quantizer(logits)
    frequency = (output.hard > 0).float().mean()
    assert abs(float(frequency) - 0.7) < 0.02


def _small_autoencoder(bits: int = 3) -> BinaryImageAutoencoder:
    return BinaryImageAutoencoder(
        bits=bits,
        hidden=8,
        downsample_layers=1,
        residual_layers=1,
        residual_hidden=4,
        stochastic=False,
        grouping="per_channel",
    )


def test_threshold_scale_ema_updates_and_freezes():
    model = _small_autoencoder(bits=2)
    model.train()
    logits = torch.tensor([[[[1.0, -3.0]], [[2.0, -4.0]]]])

    scales = model.threshold_scales(logits, ema_decay=0.9, scale_min=1e-3, update=True).flatten()
    expected = logits.float().square().mean(dim=(0, 2, 3)).sqrt()

    assert bool(model.threshold_scale_ema_initialized)
    assert torch.allclose(scales, expected)

    frozen = model.threshold_scale_ema.clone()
    model.eval()
    model.threshold_scales(torch.full_like(logits, 100.0), ema_decay=0.9, scale_min=1e-3, update=False)
    model.threshold_scales(torch.full_like(logits, 100.0), ema_decay=0.9, scale_min=1e-3, update=True)

    assert torch.equal(model.threshold_scale_ema, frozen)


def test_threshold_scale_ema_clamps_to_positive_minimum():
    model = _small_autoencoder(bits=2)
    model.train()
    scales = model.threshold_scales(torch.zeros(1, 2, 2, 2), ema_decay=0.9, scale_min=0.25, update=True)

    assert torch.equal(scales.flatten(), torch.full((2,), 0.25))
    assert torch.equal(model.threshold_scale_ema, torch.full((2,), 0.25))


def test_threshold_scale_ema_is_checkpoint_state():
    model = _small_autoencoder(bits=2)
    model.train()
    logits = torch.tensor([[[[1.0, -3.0]], [[2.0, -4.0]]]])
    model.threshold_scales(logits, ema_decay=0.9, scale_min=1e-3, update=True)

    restored = _small_autoencoder(bits=2)
    restored.load_state_dict(model.state_dict())

    assert torch.equal(restored.threshold_scale_ema, model.threshold_scale_ema)
    assert torch.equal(restored.threshold_scale_ema_initialized, model.threshold_scale_ema_initialized)
