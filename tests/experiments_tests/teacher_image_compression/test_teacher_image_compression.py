import importlib
import inspect
import math
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from relational_compression.experiments.teacher_image_compression import metrics as teacher_metrics
from relational_compression.experiments.teacher_image_compression.collisions import (
    SignedPartitionLoss,
    SignedTeacherGraph,
    same_code_probability,
    signed_partition_loss,
    signed_teacher_graph,
    soft_bit_probabilities,
)
from relational_compression.experiments.teacher_image_compression.decoder_stage import (
    _config_dict_for_decoder_checkpoint,
    _config_dict_for_representation_checkpoint,
    _log_decoder_reconstruction_images,
    _make_input_reconstruction_grid,
    reconstruction_from_hard_codes,
    train_decoder_stage,
)
from relational_compression.experiments.teacher_image_compression.metrics import (
    evaluate_representation,
    hard_code_metrics,
    hard_collision_rate,
    weighted_hard_collision_rate,
)
from relational_compression.experiments.teacher_image_compression.models import (
    DINO_VITS8_REPO,
    DINO_VITS8_SOURCE_REVISION,
    BinaryPatchEncoder,
    DinoVitS8PatchTeacher,
    make_decoder,
    torch_hub_repo_cache_name,
)
from relational_compression.experiments.teacher_image_compression.run_scripts import run_flowers_exp
from relational_compression.experiments.teacher_image_compression.run_scripts.run_flowers_exp import _run_checkpoint_dir
from relational_compression.experiments.teacher_image_compression.sampling import gather_tokens, sample_token_indices


class FakeDinoBackbone(nn.Module):
    def __init__(self, embedding_dim: int = 6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.embedding_dim = embedding_dim
        self.last_input: torch.Tensor | None = None

    def get_intermediate_layers(self, images: torch.Tensor, n: int = 1) -> list[torch.Tensor]:
        del n
        self.last_input = images.detach().clone()
        batch, _, height, width = images.shape
        patch_count = (height // 8) * (width // 8)
        values = torch.arange(
            batch * (patch_count + 1) * self.embedding_dim,
            device=images.device,
            dtype=images.dtype,
        ).reshape(batch, patch_count + 1, self.embedding_dim)
        return [values * self.weight]


class ConstantGridTeacher(nn.Module):
    def __init__(self, tokens: torch.Tensor | None = None, embedding_dim: int = 3) -> None:
        super().__init__()
        self.tokens = tokens
        self.embedding_dim = embedding_dim

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.tokens is not None:
            return self.tokens.to(device=images.device, dtype=images.dtype)

        rows = torch.linspace(-1.0, 1.0, 4, device=images.device, dtype=images.dtype)
        cols = torch.linspace(-1.0, 1.0, 4, device=images.device, dtype=images.dtype)
        yy, xx = torch.meshgrid(rows, cols, indexing="ij")
        grid = torch.stack((yy, xx, yy * xx), dim=-1)[..., : self.embedding_dim]
        grid = torch.nn.functional.normalize(grid, dim=-1)
        return grid.view(1, 4, 4, self.embedding_dim).expand(images.shape[0], -1, -1, -1)


class FixedGridStudent(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits_parameter = nn.Parameter(logits)

    def forward(self, images: torch.Tensor, *, force_hard: bool = False) -> SimpleNamespace:
        del images, force_hard
        hard = torch.where(
            self.logits_parameter > 0.0,
            torch.ones_like(self.logits_parameter),
            -torch.ones_like(self.logits_parameter),
        )
        return SimpleNamespace(logits=self.logits_parameter, hard=hard)

    def hard_codes(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward(images, force_hard=True).hard


class StaticEncoder(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = nn.Parameter(logits)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.logits.expand(images.shape[0], -1, -1, -1)


class HardCodeOnlyStudent(nn.Module):
    def __init__(self, bits: int) -> None:
        super().__init__()
        self.bits = bits

    def hard_codes(self, images: torch.Tensor) -> torch.Tensor:
        shape = (images.shape[0], self.bits, images.shape[-2] // 2, images.shape[-1] // 2)
        return torch.ones(shape, device=images.device, dtype=images.dtype)


class DummyTensorBoardWriter:
    def __init__(self) -> None:
        self.images: list[tuple[str, int, torch.Tensor]] = []
        self.scalars: list[tuple[str, int]] = []
        self.flush_count = 0

    def add_image(self, tag: str, img_tensor: torch.Tensor, global_step: int) -> None:
        self.images.append((tag, global_step, img_tensor))

    def add_scalar(self, tag: str, scalar_value: object, global_step: int) -> None:
        del scalar_value
        self.scalars.append((tag, global_step))

    def flush(self) -> None:
        self.flush_count += 1


def _small_config(**overrides: Any) -> SimpleNamespace:
    values = {
        "sampled_tokens": 8,
        "prefer_cross_image_neighbors": False,
        "code_temperature": 0.25,
        "teacher_temperature": 0.1,
        "bits": 2,
        "alignment_weight": 1.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_dino_patch_teacher_extracts_normalized_32x32_grid_and_freezes_parameters() -> None:
    backbone = FakeDinoBackbone()
    teacher = DinoVitS8PatchTeacher(backbone)
    images = torch.rand(2, 3, 256, 256)

    tokens = teacher(images)

    assert tokens.shape == (2, 32, 32, 6)
    assert torch.allclose(tokens.norm(dim=-1), torch.ones(2, 32, 32), atol=1e-6)
    assert not any(parameter.requires_grad for parameter in teacher.parameters())

    teacher.train(True)
    assert not teacher.training
    assert not teacher.backbone.training


def test_dino_teacher_applies_imagenet_normalization_to_student_image() -> None:
    backbone = FakeDinoBackbone()
    teacher = DinoVitS8PatchTeacher(backbone)

    teacher(torch.zeros(1, 3, 256, 256))

    expected = -torch.tensor([0.485 / 0.229, 0.456 / 0.224, 0.406 / 0.225]).view(1, 3, 1, 1)
    assert backbone.last_input is not None
    assert torch.allclose(backbone.last_input[:, :, :1, :1], expected, atol=1e-5)


def test_binary_patch_encoder_matches_teacher_grid_for_256_inputs() -> None:
    student = BinaryPatchEncoder(
        bits=4,
        hidden=8,
        downsample_layers=3,
        residual_layers=1,
        residual_hidden=4,
    )

    output = student(torch.rand(2, 3, 256, 256))

    assert output.logits.shape == (2, 4, 32, 32)
    assert output.hard.shape == output.logits.shape
    assert set(output.hard.unique().tolist()) <= {-1.0, 1.0}


def test_teacher_student_has_no_mean_group_ema_or_separation_state() -> None:
    student = BinaryPatchEncoder(
        bits=4,
        hidden=8,
        downsample_layers=1,
        residual_layers=1,
        residual_hidden=4,
    )

    assert not hasattr(student, "quantizer")
    assert not hasattr(student, "threshold_scale_ema")
    assert "separation" not in run_flowers_exp._representation_batch_loss.__code__.co_names


def test_hard_code_is_exact_zero_threshold_sign_coding() -> None:
    logits = torch.tensor([[[[-1.0, 0.0], [0.2, 3.0]]]])
    student = BinaryPatchEncoder(
        bits=1,
        hidden=4,
        downsample_layers=1,
        residual_layers=1,
        residual_hidden=4,
    )
    student.encoder = StaticEncoder(logits)

    output = student(torch.rand(1, 3, 2, 2))

    expected = torch.tensor([[[[-1.0, -1.0], [1.0, 1.0]]]])
    assert torch.equal(output.hard, expected)


def test_sampling_is_bounded_and_gathers_corresponding_tokens() -> None:
    image_indices, spatial_indices = sample_token_indices(
        batch_size=4,
        grid_h=32,
        grid_w=32,
        max_tokens=17,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(3),
    )
    assert image_indices.shape == (17,)
    assert spatial_indices.shape == (17,)

    teacher = torch.randn(4, 32, 32, 5)
    logits = torch.randn(4, 3, 32, 32)
    hard = torch.where(logits > 0.0, torch.ones_like(logits), -torch.ones_like(logits))
    sample = gather_tokens(
        teacher_grid=teacher,
        logits=logits,
        hard=hard,
        image_indices=image_indices,
        spatial_indices=spatial_indices,
    )

    assert sample.teacher_tokens.shape == (17, 5)
    assert sample.logits.shape == (17, 3)
    assert sample.hard.shape == (17, 3)


def test_signed_teacher_graph_excludes_self_pairs_and_respects_cross_image_mask() -> None:
    teacher_tokens = torch.nn.functional.normalize(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
        dim=-1,
    )
    image_indices = torch.tensor([0, 0, 1, 1])

    graph = signed_teacher_graph(
        teacher_tokens,
        image_indices,
        teacher_temperature=0.1,
        prefer_cross_image=True,
    )
    unrestricted = signed_teacher_graph(
        teacher_tokens,
        image_indices,
        teacher_temperature=0.1,
        prefer_cross_image=False,
    )

    assert torch.all(graph.pairs[:, 0] != graph.pairs[:, 1])
    assert torch.all(image_indices[graph.pairs[:, 0]] != image_indices[graph.pairs[:, 1]])
    assert unrestricted.pairs.shape[0] == 4 * 3


def test_signed_teacher_graph_rows_are_normalized_over_valid_candidates() -> None:
    teacher_tokens = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [-1.0, 0.0]]),
        dim=-1,
    )
    image_indices = torch.tensor([0, 0, 1, 1])

    graph = signed_teacher_graph(
        teacher_tokens,
        image_indices,
        teacher_temperature=0.25,
        prefer_cross_image=True,
    )

    for row in torch.unique(graph.pairs[:, 0]):
        mask = graph.pairs[:, 0] == row
        assert torch.allclose(graph.teacher_probabilities[mask].sum(), torch.tensor(1.0), atol=1e-6)
        assert torch.allclose(graph.neutral_probabilities[mask].sum(), torch.tensor(1.0), atol=1e-6)
        assert torch.allclose(graph.positive_weights[mask].sum(), graph.negative_weights[mask].sum(), atol=1e-6)


def test_empty_cross_image_signed_graph_and_loss_do_not_use_tensor_any(monkeypatch) -> None:
    monkeypatch.setattr(torch, "any", lambda *args, **kwargs: pytest.fail("torch.any should not be called"))
    teacher_tokens = torch.nn.functional.normalize(torch.randn(4, 3), dim=-1)
    image_indices = torch.zeros(4, dtype=torch.long)
    logits = torch.randn(4, 2, requires_grad=True)

    graph = signed_teacher_graph(
        teacher_tokens,
        image_indices,
        teacher_temperature=0.1,
        prefer_cross_image=True,
    )
    loss = signed_partition_loss(soft_bit_probabilities(logits, 0.25), graph, bits=2)
    loss.loss.backward()

    assert graph.pairs.shape == (0, 2)
    assert loss.loss == 0.0
    assert logits.grad is not None
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_signed_graph_and_loss_avoid_reviewed_cuda_sync_patterns() -> None:
    assert "torch.any" not in inspect.getsource(signed_teacher_graph)
    loss_source = inspect.getsource(signed_partition_loss)
    assert "float(" not in loss_source
    assert ".item()" not in loss_source


def test_signed_partition_loss_handles_zero_weight_mass_without_python_mass_branching() -> None:
    probabilities = torch.rand(3, 2, requires_grad=True)
    graph = SignedTeacherGraph(
        pairs=torch.tensor([[0, 1], [1, 2]]),
        similarities=torch.tensor([0.5, -0.5]),
        teacher_probabilities=torch.tensor([0.5, 0.5]),
        neutral_probabilities=torch.tensor([0.5, 0.5]),
        positive_weights=torch.zeros(2),
        negative_weights=torch.zeros(2),
    )

    loss = signed_partition_loss(probabilities, graph, bits=2)
    loss.loss.backward()

    assert loss.loss == 0.0
    assert loss.positive_loss == 0.0
    assert loss.negative_loss == 0.0
    assert probabilities.grad is not None
    assert torch.equal(probabilities.grad, torch.zeros_like(probabilities))


def test_same_bucket_probability_matches_factorized_formula() -> None:
    probabilities = torch.tensor([[0.8, 0.3], [0.6, 0.1]])
    pairs = torch.tensor([[0, 1]])

    collision = same_code_probability(probabilities, pairs)

    expected = (0.8 * 0.6 + 0.2 * 0.4) * (0.3 * 0.1 + 0.7 * 0.9)
    assert torch.allclose(collision, torch.tensor([expected]), atol=1e-6)


def test_tensorized_signed_loss_matches_reference_formula() -> None:
    probabilities = torch.tensor(
        [
            [0.8, 0.3],
            [0.6, 0.1],
            [0.2, 0.9],
        ],
        requires_grad=True,
    )
    graph = SignedTeacherGraph(
        pairs=torch.tensor([[0, 1], [0, 2], [1, 2]]),
        similarities=torch.tensor([0.7, -0.4, 0.1]),
        teacher_probabilities=torch.tensor([0.7, 0.1, 0.2]),
        neutral_probabilities=torch.full((3,), 1.0 / 3.0),
        positive_weights=torch.tensor([0.5, 0.0, 0.25]),
        negative_weights=torch.tensor([0.0, 0.4, 0.0]),
    )

    loss = signed_partition_loss(probabilities, graph, bits=2)
    same = same_code_probability(probabilities, graph.pairs)
    prior = 2.0**-2
    margin = torch.logit(same.clamp(torch.finfo(same.dtype).eps, 1.0 - torch.finfo(same.dtype).eps)) - math.log(
        prior / (1.0 - prior)
    )
    expected_positive = (graph.positive_weights * torch.nn.functional.softplus(-margin)).sum()
    expected_positive = expected_positive / graph.positive_weights.sum()
    expected_negative = (graph.negative_weights * torch.nn.functional.softplus(margin)).sum()
    expected_negative = expected_negative / graph.negative_weights.sum()
    expected = 0.5 * (expected_positive + expected_negative)

    assert torch.allclose(loss.positive_loss, expected_positive, atol=1e-6)
    assert torch.allclose(loss.negative_loss, expected_negative, atol=1e-6)
    assert torch.allclose(loss.loss, expected, atol=1e-6)


def test_prior_centered_signed_loss_is_finite_and_differentiable() -> None:
    logits = torch.tensor([[2.0, -1.0], [1.5, -1.2], [-2.0, 2.0], [0.5, 0.5]], requires_grad=True)
    teacher_tokens = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]]),
        dim=-1,
    )
    graph = signed_teacher_graph(
        teacher_tokens,
        torch.tensor([0, 0, 1, 1]),
        teacher_temperature=0.2,
        prefer_cross_image=False,
    )

    probabilities = soft_bit_probabilities(logits, 0.25)
    loss = signed_partition_loss(probabilities, graph, bits=2)
    loss.loss.backward()

    assert torch.isfinite(loss.loss)
    assert torch.isfinite(loss.positive_loss)
    assert torch.isfinite(loss.negative_loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_signed_partition_loss_rejects_incompatible_collision_floor() -> None:
    probabilities = torch.full((2, 16), 0.5, dtype=torch.float16)
    graph = SignedTeacherGraph(
        pairs=torch.tensor([[0, 1]]),
        similarities=torch.tensor([0.0]),
        teacher_probabilities=torch.tensor([1.0]),
        neutral_probabilities=torch.tensor([0.0]),
        positive_weights=torch.tensor([1.0]),
        negative_weights=torch.tensor([0.0]),
    )

    with pytest.raises(
        ValueError,
        match="collision clipping floor must be below the neutral same-code probability",
    ):
        signed_partition_loss(probabilities, graph, bits=16)


def test_signed_partition_loss_returns_zero_for_empty_pairs_before_dtype_guard() -> None:
    probabilities = torch.full((2, 16), 0.5, dtype=torch.float16)
    empty = torch.empty(0)
    graph = SignedTeacherGraph(
        pairs=torch.empty((0, 2), dtype=torch.long),
        similarities=empty,
        teacher_probabilities=empty,
        neutral_probabilities=empty,
        positive_weights=empty,
        negative_weights=empty,
    )

    loss = signed_partition_loss(probabilities, graph, bits=16)

    assert torch.isfinite(loss.loss)
    assert loss.loss == 0.0
    assert loss.positive_loss == 0.0
    assert loss.negative_loss == 0.0
    assert loss.same_probability.numel() == 0


def test_positive_teacher_edges_prefer_higher_collision() -> None:
    graph = SignedTeacherGraph(
        pairs=torch.tensor([[0, 1]]),
        similarities=torch.tensor([1.0]),
        teacher_probabilities=torch.tensor([1.0]),
        neutral_probabilities=torch.tensor([0.0]),
        positive_weights=torch.tensor([1.0]),
        negative_weights=torch.tensor([0.0]),
    )
    matching = soft_bit_probabilities(torch.tensor([[4.0, 4.0], [4.0, 4.0]]), 0.25)
    separated = soft_bit_probabilities(torch.tensor([[4.0, 4.0], [-4.0, -4.0]]), 0.25)

    matching_loss = signed_partition_loss(matching, graph, bits=2)
    separated_loss = signed_partition_loss(separated, graph, bits=2)

    assert matching_loss.loss < separated_loss.loss


def test_negative_teacher_edges_prefer_lower_collision() -> None:
    graph = SignedTeacherGraph(
        pairs=torch.tensor([[0, 1]]),
        similarities=torch.tensor([-1.0]),
        teacher_probabilities=torch.tensor([0.0]),
        neutral_probabilities=torch.tensor([1.0]),
        positive_weights=torch.tensor([0.0]),
        negative_weights=torch.tensor([1.0]),
    )
    matching = soft_bit_probabilities(torch.tensor([[4.0, 4.0], [4.0, 4.0]]), 0.25)
    separated = soft_bit_probabilities(torch.tensor([[4.0, 4.0], [-4.0, -4.0]]), 0.25)

    matching_loss = signed_partition_loss(matching, graph, bits=2)
    separated_loss = signed_partition_loss(separated, graph, bits=2)

    assert separated_loss.loss < matching_loss.loss


def test_universal_one_code_collapse_incurs_negative_edge_loss() -> None:
    graph = SignedTeacherGraph(
        pairs=torch.tensor([[0, 1], [0, 2]]),
        similarities=torch.tensor([1.0, -1.0]),
        teacher_probabilities=torch.tensor([1.0, 0.0]),
        neutral_probabilities=torch.tensor([0.5, 0.5]),
        positive_weights=torch.tensor([1.0, 0.0]),
        negative_weights=torch.tensor([0.0, 1.0]),
    )
    collapsed = soft_bit_probabilities(torch.full((3, 2), 4.0), 0.25)
    separated = soft_bit_probabilities(torch.tensor([[4.0, 4.0], [4.0, 4.0], [-4.0, -4.0]]), 0.25)

    collapsed_loss = signed_partition_loss(collapsed, graph, bits=2)
    separated_loss = signed_partition_loss(separated, graph, bits=2)

    assert collapsed_loss.negative_loss > 1.0
    assert separated_loss.loss < collapsed_loss.loss


def test_representation_loss_uses_signed_teacher_objective() -> None:
    logits = torch.randn(2, 2, 4, 4, requires_grad=True)
    student = FixedGridStudent(logits)
    teacher = ConstantGridTeacher()

    values = run_flowers_exp._representation_batch_loss(
        student=student,
        teacher=teacher,
        images=torch.rand(2, 3, 32, 32),
        config=_small_config(sampled_tokens=12, prefer_cross_image_neighbors=True),
    )
    values["loss"].backward()

    assert torch.isfinite(values["loss"])
    assert student.logits_parameter.grad is not None
    assert torch.isfinite(student.logits_parameter.grad).all()
    assert "teacher_loss" in values
    assert "teacher_positive_loss" in values
    assert "teacher_negative_loss" in values
    assert "concentration" not in values
    assert "separation_d2" not in values


def test_teacher_similarity_diagnostic_panel_shape_and_range() -> None:
    image = torch.rand(3, 32, 32)
    teacher_grid = torch.nn.functional.normalize(torch.randn(4, 4, 6), dim=-1)

    panel = run_flowers_exp._teacher_similarity_diagnostic_panel(
        image=image,
        teacher_grid=teacher_grid,
        heatmap_size=64,
    )

    assert panel.shape == (3, 64, 130)
    assert float(panel.min()) >= 0.0
    assert float(panel.max()) <= 1.0


def test_representation_loss_can_return_similarity_diagnostics() -> None:
    logits = torch.randn(1, 2, 4, 4, requires_grad=True)
    student = FixedGridStudent(logits)
    teacher = ConstantGridTeacher()

    values = run_flowers_exp._representation_batch_loss(
        student=student,
        teacher=teacher,
        images=torch.rand(1, 3, 32, 32),
        config=_small_config(sampled_tokens=4),
        return_diagnostics=True,
    )

    assert values["diagnostic_image"].shape == (3, 32, 32)
    assert values["diagnostic_teacher_grid"].shape == (4, 4, 3)


def test_hard_metrics_and_pair_collision_rates() -> None:
    hard = torch.tensor(
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [-1.0, -1.0],
            [1.0, -1.0],
        ]
    )
    metrics = hard_code_metrics(hard)

    assert metrics.active_codes == 3
    assert metrics.bit_occupancy_mean == 0.625
    assert math.isclose(metrics.empirical_collision, 0.375)
    assert math.isclose(metrics.hard_h2, -math.log(0.375), rel_tol=1e-6)
    assert hard_collision_rate(hard, torch.tensor([[0, 1], [0, 2]])) == 0.5
    assert weighted_hard_collision_rate(hard, torch.tensor([[0, 1], [0, 2]]), torch.tensor([2.0, 1.0])) == 2.0 / 3.0


def test_evaluation_reports_signed_partition_metrics() -> None:
    logits = torch.tensor(
        [
            [[[2.0, 2.0], [-2.0, -2.0]]],
            [[[2.0, -2.0], [2.0, -2.0]]],
        ]
    )
    teacher_tokens = torch.nn.functional.normalize(
        torch.tensor(
            [
                [[[1.0, 0.0], [0.8, 0.2]], [[0.0, 1.0], [-1.0, 0.0]]],
                [[[1.0, 0.0], [0.7, 0.3]], [[0.0, 1.0], [-1.0, 0.0]]],
            ]
        ),
        dim=-1,
    )
    student = FixedGridStudent(logits)
    loader = [torch.rand(2, 3, 16, 16)]

    metrics = evaluate_representation(
        student=student,
        teacher=ConstantGridTeacher(tokens=teacher_tokens),
        loader=loader,
        device=torch.device("cpu"),
        bits=1,
        sampled_tokens=8,
        code_temperature=0.25,
        teacher_temperature=0.1,
        prefer_cross_image=True,
        max_batches=None,
        seed=7,
    )

    assert math.isfinite(metrics["teacher_positive_hard_collision_rate"])
    assert math.isfinite(metrics["teacher_negative_hard_collision_rate"])
    assert math.isclose(
        metrics["teacher_partition_gap"],
        metrics["teacher_positive_hard_collision_rate"] - metrics["teacher_negative_hard_collision_rate"],
        rel_tol=1e-6,
    )
    assert math.isfinite(metrics["teacher_positive_collision_enrichment"])
    assert math.isfinite(metrics["teacher_negative_collision_suppression"])


def test_weighted_teacher_cosine_summary_uses_edge_weights() -> None:
    summary = teacher_metrics._weighted_summary(
        "teacher_positive_cosine",
        [torch.tensor([0.0, 1.0])],
        [torch.tensor([1.0, 3.0])],
    )

    assert math.isclose(summary["teacher_positive_cosine_mean"], 0.75, rel_tol=1e-6)
    assert math.isclose(summary["teacher_positive_cosine_std"], math.sqrt(0.1875), rel_tol=1e-6)
    assert summary["teacher_positive_cosine_min"] == 0.0
    assert summary["teacher_positive_cosine_max"] == 1.0


def test_evaluation_aggregates_teacher_losses_by_signed_weight_mass(monkeypatch) -> None:
    logits = torch.tensor(
        [
            [[[2.0, 2.0], [-2.0, -2.0]]],
            [[[2.0, -2.0], [2.0, -2.0]]],
        ]
    )
    student = FixedGridStudent(logits)
    teacher_tokens = torch.nn.functional.normalize(torch.randn(2, 2, 2, 3), dim=-1)
    loader = [torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)]
    losses = iter(((1.0, 2.0, 10.0, 1.0), (3.0, 6.0, 20.0, 3.0)))

    def fake_signed_partition_loss(
        soft_bit_probabilities: torch.Tensor, graph: SignedTeacherGraph, *, bits: int
    ) -> SignedPartitionLoss:
        del graph, bits
        positive_loss, positive_mass, negative_loss, negative_mass = next(losses)
        zero = soft_bit_probabilities.sum() * 0.0
        return SignedPartitionLoss(
            loss=zero,
            positive_loss=zero + positive_loss,
            negative_loss=zero + negative_loss,
            positive_weight_mass=torch.tensor(positive_mass),
            negative_weight_mass=torch.tensor(negative_mass),
            same_probability=torch.empty(0),
            positive_same_probability=torch.empty(0),
            negative_same_probability=torch.empty(0),
        )

    monkeypatch.setattr(teacher_metrics, "signed_partition_loss", fake_signed_partition_loss)

    metrics = evaluate_representation(
        student=student,
        teacher=ConstantGridTeacher(tokens=teacher_tokens),
        loader=loader,
        device=torch.device("cpu"),
        bits=1,
        sampled_tokens=8,
        code_temperature=0.25,
        teacher_temperature=0.1,
        prefer_cross_image=True,
        max_batches=None,
        seed=7,
    )

    assert math.isclose(metrics["teacher_positive_loss"], (1.0 * 2.0 + 3.0 * 6.0) / 8.0)
    assert math.isclose(metrics["teacher_negative_loss"], (10.0 * 1.0 + 20.0 * 3.0) / 4.0)
    assert math.isclose(metrics["teacher_loss"], 10.0)


def test_fixed_validation_sampling_is_reproducible() -> None:
    logits = torch.randn(2, 2, 4, 4)
    student = FixedGridStudent(logits)
    loader = [torch.rand(2, 3, 32, 32)]

    first = evaluate_representation(
        student=student,
        teacher=ConstantGridTeacher(),
        loader=loader,
        device=torch.device("cpu"),
        bits=2,
        sampled_tokens=12,
        code_temperature=0.25,
        teacher_temperature=0.1,
        prefer_cross_image=True,
        max_batches=None,
        seed=123,
    )
    second = evaluate_representation(
        student=student,
        teacher=ConstantGridTeacher(),
        loader=loader,
        device=torch.device("cpu"),
        bits=2,
        sampled_tokens=12,
        code_temperature=0.25,
        teacher_temperature=0.1,
        prefer_cross_image=True,
        max_batches=None,
        seed=123,
    )

    assert first == second


def test_checkpoint_selection_uses_validation_teacher_loss() -> None:
    metrics = {
        "teacher_loss": 0.4,
        "teacher_partition_gap": 0.25,
        "teacher_positive_collision_enrichment": 99.0,
    }

    assert run_flowers_exp._checkpoint_score(metrics) == 0.4
    assert run_flowers_exp._checkpoint_score({"teacher_loss": float("nan")}) == float("inf")


def test_representation_checkpoint_remains_resumable_after_decoder_config_changes() -> None:
    base = SimpleNamespace(
        task_model_name="teacher_image_compression",
        dataset_name="Flowers102",
        dataset_version="1.0",
        image_size=256,
        batch_size=32,
        seed=1337,
        learning_rate=5e-4,
        min_learning_rate=1e-6,
        learning_rate_warmup_steps=100,
        learning_rate_warmup_start_factor=0.1,
        weight_decay=0.0,
        bits=16,
        hidden=128,
        downsample_layers=3,
        residual_layers=2,
        residual_hidden=64,
        alignment_weight=1.0,
        code_temperature=0.25,
        teacher_temperature=0.1,
        sampled_tokens=512,
        eval_sampled_tokens=1024,
        prefer_cross_image_neighbors=True,
        validation_sampling_seed=None,
        test_sampling_seed=None,
        teacher_backend="fixed_random_patch",
        teacher_repo="unused",
        teacher_model_name="unused",
        teacher_checkpoint_url="unused",
        fixed_random_teacher_embedding_dim=64,
        max_train_batches=None,
        max_val_batches=None,
        decoder_num_epochs=0,
        decoder_learning_rate=5e-4,
        decoder_weight_decay=0.0,
        unique_postfix="first",
        experiment_name="runtime-a",
        device="cuda",
    )
    changed_runtime = SimpleNamespace(
        **{**vars(base), "unique_postfix": "second", "experiment_name": "runtime-b", "device": "cpu"}
    )
    changed_decoder = SimpleNamespace(
        **{
            **vars(base),
            "decoder_num_epochs": 10,
            "decoder_learning_rate": 1e-4,
            "decoder_weight_decay": 1e-2,
        }
    )
    changed_objective = SimpleNamespace(**{**vars(base), "teacher_temperature": 0.2})

    saved_checkpoint = {"config": _config_dict_for_representation_checkpoint(base), "completed_epoch": 1}

    assert saved_checkpoint["config"] == _config_dict_for_representation_checkpoint(changed_runtime)
    assert saved_checkpoint["config"] == _config_dict_for_representation_checkpoint(changed_decoder)
    assert _config_dict_for_representation_checkpoint(base) != _config_dict_for_representation_checkpoint(
        changed_objective
    )


def test_decoder_checkpoint_config_detects_incompatible_decoder_settings() -> None:
    base = SimpleNamespace(
        task_model_name="teacher_image_compression",
        dataset_name="Flowers102",
        dataset_version="1.0",
        image_size=256,
        batch_size=32,
        seed=1337,
        bits=16,
        hidden=128,
        downsample_layers=3,
        residual_layers=2,
        residual_hidden=64,
        decoder_num_epochs=5,
        decoder_learning_rate=5e-4,
        decoder_weight_decay=0.0,
        unique_postfix="first",
        experiment_name="runtime-a",
        device="cuda",
    )
    changed_runtime = SimpleNamespace(
        **{**vars(base), "unique_postfix": "second", "experiment_name": "runtime-b", "device": "cpu"}
    )
    changed_decoder = SimpleNamespace(**{**vars(base), "decoder_learning_rate": 1e-4})

    assert _config_dict_for_decoder_checkpoint(base) == _config_dict_for_decoder_checkpoint(changed_runtime)
    assert _config_dict_for_decoder_checkpoint(base) != _config_dict_for_decoder_checkpoint(changed_decoder)


def test_checkpoint_paths_are_experiment_local(tmp_path) -> None:
    experiment_root = tmp_path / "teacher_image_experiment"

    path = _run_checkpoint_dir(experiment_root=experiment_root, run_name="run_00_deterministic")

    assert path == experiment_root / "checkpoints" / "run_00_deterministic"


def test_decoder_reconstruction_gradients_do_not_reach_encoder() -> None:
    config = SimpleNamespace(
        bits=3,
        hidden=8,
        downsample_layers=1,
        residual_layers=1,
        residual_hidden=4,
    )
    student = BinaryPatchEncoder(
        bits=3,
        hidden=8,
        downsample_layers=1,
        residual_layers=1,
        residual_hidden=4,
    )
    decoder = make_decoder(config)
    images = torch.rand(2, 3, 32, 32)

    reconstruction = reconstruction_from_hard_codes(student, decoder, images)
    reconstruction.mean().backward()

    assert all(parameter.grad is None for parameter in student.parameters())
    assert any(parameter.grad is not None for parameter in decoder.parameters())


def test_decoder_input_reconstruction_grid_uses_first_pair() -> None:
    images = torch.full((2, 3, 4, 4), -1.0)
    reconstructions = torch.full((2, 3, 4, 4), 2.0)

    grid = _make_input_reconstruction_grid(images, reconstructions)

    assert grid.shape == (3, 8, 14)
    assert float(grid.min()) == 0.0
    assert float(grid.max()) == 1.0


def test_decoder_reconstruction_image_logging_restores_model_state() -> None:
    student = HardCodeOnlyStudent(bits=2)
    decoder = make_decoder(SimpleNamespace(bits=2, hidden=4, downsample_layers=1, residual_layers=1, residual_hidden=4))
    student.train()
    decoder.train()
    writer = DummyTensorBoardWriter()

    _log_decoder_reconstruction_images(
        writer=writer,
        student=student,
        decoder=decoder,
        loader=[torch.rand(2, 3, 16, 16)],
        device=torch.device("cpu"),
        epoch=7,
    )

    assert student.training
    assert decoder.training
    assert [(tag, step) for tag, step, _ in writer.images] == [("decoder/validate_input_reconstruction", 7)]


def test_decoder_stage_logs_validation_reconstruction_images(tmp_path) -> None:
    config = SimpleNamespace(
        task_model_name="teacher_image_compression",
        dataset_name="Flowers102",
        dataset_version="1.0",
        image_size=16,
        batch_size=2,
        seed=1,
        min_learning_rate=0.0,
        learning_rate_warmup_steps=0,
        learning_rate_warmup_start_factor=0.1,
        bits=2,
        hidden=4,
        downsample_layers=1,
        residual_layers=1,
        residual_hidden=4,
        max_train_batches=1,
        max_val_batches=1,
        decoder_num_epochs=1,
        decoder_learning_rate=1e-3,
        decoder_weight_decay=0.0,
        decoder_reconstruction_log_epochs=1,
        tensorboard_log_steps=1,
    )
    writer = DummyTensorBoardWriter()
    images = torch.rand(2, 3, 16, 16)

    result = train_decoder_stage(
        student=HardCodeOnlyStudent(bits=2),
        config=config,
        run_dir=tmp_path / "run",
        checkpoint_dir=tmp_path / "checkpoints" / "decoder",
        train_loader=[images],
        valid_loader=[images],
        test_loader=[images],
        device=torch.device("cpu"),
        writer=writer,
    )

    assert result["best_epoch"] == 1
    assert ("decoder/validate_input_reconstruction", 1) in [(tag, step) for tag, step, _ in writer.images]
    assert writer.flush_count == 1


def test_baseline_config_imports_without_network(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RELCO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RELCO_OUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("RELCO_CHECKPOINT_DIR", str(tmp_path / "checkpoints"))

    module = importlib.import_module(
        "relational_compression.experiments.teacher_image_compression.configs.flowers102_dino_vits8"
    )
    module = importlib.reload(module)

    assert module.teacher_backend == "dino_vits8"
    assert module.teacher_repo == DINO_VITS8_REPO
    assert DINO_VITS8_SOURCE_REVISION in module.teacher_repo
    assert module.teacher_model_name == "dino_vits8"
    assert module.bits == 16
    assert module.image_size == 256
    assert module.code_temperature == 0.25
    assert module.teacher_temperature == 0.1
    assert not hasattr(module, "separation_weight")
    assert not hasattr(module, "concentration_weight")
    assert module.teacher_cache_dir == tmp_path / "out" / "model_cache" / "teacher_image_compression" / "dino_vits8"


def test_dino_teacher_loader_no_download_uses_pinned_torch_hub_cache(monkeypatch, tmp_path) -> None:
    cache_dir = tmp_path / "dino"
    cache_dir.joinpath(torch_hub_repo_cache_name(DINO_VITS8_REPO)).mkdir(parents=True)
    cache_dir.joinpath("checkpoints").mkdir()
    cache_dir.joinpath("checkpoints", "dino_deitsmall8_pretrain.pth").write_bytes(b"checkpoint")
    observed: dict[str, str] = {}

    def fake_load(repo_or_dir: str, model: str, **kwargs: Any) -> FakeDinoBackbone:
        del kwargs
        observed["repo"] = repo_or_dir
        observed["model_name"] = model
        return FakeDinoBackbone()

    monkeypatch.setattr(torch.hub, "load", fake_load)

    teacher = DinoVitS8PatchTeacher.from_torch_hub(cache_dir=cache_dir, allow_download=False)

    assert observed == {"repo": DINO_VITS8_REPO, "model_name": "dino_vits8"}
    assert isinstance(teacher, DinoVitS8PatchTeacher)


def test_dino_teacher_loader_can_be_blocked_before_network(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: pytest.fail("torch.hub.load should not be called"))

    with pytest.raises(FileNotFoundError):
        DinoVitS8PatchTeacher.from_torch_hub(cache_dir=tmp_path / "dino", allow_download=False)
