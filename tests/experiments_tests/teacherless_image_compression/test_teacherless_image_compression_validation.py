from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader


class _ValidationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.quantizer = nn.Identity()

    def forward(self, images: torch.Tensor, force_hard: bool = False) -> SimpleNamespace:
        batch_size = images.shape[0]
        value = (1.0 if batch_size == 2 else 3.0) if force_hard else (2.0 if batch_size == 2 else 4.0)
        quantizer = SimpleNamespace(
            hard=torch.zeros(batch_size, 1, 1, 1),
            probabilities=torch.full((batch_size, 1, 1, 1), 0.5),
            relaxed=torch.zeros(batch_size, 1, 1, 1),
        )
        return SimpleNamespace(reconstruction=torch.full_like(images, value), quantizer=quantizer)


def test_validate_weights_batch_metrics_by_batch_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RELCO_DATA_DIR", "/tmp")
    monkeypatch.setenv("RELCO_OUT_DIR", "/tmp")
    from relational_compression.experiments.teacherless_image_compression import run

    loader = DataLoader(torch.zeros(3, 1, 1, 1), batch_size=2, shuffle=False)
    monkeypatch.setattr(run, "_batch_ms_ssim", lambda prediction, target: 0.2 if target.shape[0] == 2 else 0.8)

    metrics = run._validate(
        _ValidationModel(),
        loader,
        torch.device("cpu"),
        SimpleNamespace(diagnostic_pair_samples=1),
        max_batches=None,
    )

    assert metrics["hard_mse"] == pytest.approx((2 * 1.0 + 9.0) / 3)
    assert metrics["relaxed_mse"] == pytest.approx((2 * 4.0 + 16.0) / 3)
    assert metrics["ms_ssim"] == pytest.approx((2 * 0.2 + 0.8) / 3)
