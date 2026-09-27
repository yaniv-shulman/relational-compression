"""Experiment support code for relational compression studies."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from relational_compression.experiments.transductive_relational_distortion.objectives import (
    SoftObjectiveResult,
    soft_objective,
)
from relational_compression.experiments.transductive_relational_distortion.source_geometry import SourceGeometry


@dataclass(frozen=True)
class RestartSummary:
    """Store diagnostics for one partition-optimization restart."""

    restart_index: int
    init_kind: str
    seed: int
    best_loss: float
    best_step: int
    best_own_distortion: float
    best_marginal_d2: float
    best_soft_k_eff: float
    best_assignment_confidence: float
    best_assignment_entropy: float
    final_loss: float
    final_own_distortion: float
    final_marginal_d2: float
    final_soft_k_eff: float
    assignment_confidence: float
    assignment_entropy: float
    finite: bool
    encountered_nonfinite: bool


@dataclass(frozen=True)
class OptimizedPartition:
    """Store the selected partition and its optimization diagnostics."""

    criterion: str
    lambda_org: float
    selected_restart: int
    selected_seed: int
    logits: Tensor
    probabilities: Tensor
    assignments: Tensor
    soft_result: SoftObjectiveResult
    restart_summaries: list[RestartSummary]


@dataclass(frozen=True)
class InitialLogitSpec:
    """Describe a named initial-logit construction."""

    init_kind: str
    seed: int
    logits: Tensor


def _seeded_logits(
    num_nodes: int,
    num_partitions: int,
    *,
    seed: int,
    init_scale: float,
    device: torch.device,
    collapse_bias: float | None = None,
) -> Tensor:
    """Create seeded initial partition logits."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    logits = float(init_scale) * torch.randn(num_nodes, num_partitions, generator=generator)
    if collapse_bias is not None:
        logits[:, 0] += float(collapse_bias)
    return logits.to(device=device).requires_grad_(True)


def _float_result(value: Tensor) -> float:
    """Convert a scalar tensor result to a Python float."""
    return float(value.detach().cpu())


def _restart_summary(
    *,
    restart_index: int,
    init_kind: str,
    seed: int,
    best_result: SoftObjectiveResult,
    best_step: int,
    final_result: SoftObjectiveResult,
    encountered_nonfinite: bool,
) -> RestartSummary:
    """Summarize a completed partition-optimization restart."""
    best_loss = _float_result(best_result.loss) if torch.isfinite(best_result.loss) else math.inf
    final_loss = _float_result(final_result.loss) if torch.isfinite(final_result.loss) else math.inf
    return RestartSummary(
        restart_index=restart_index,
        init_kind=init_kind,
        seed=int(seed),
        best_loss=best_loss,
        best_step=int(best_step),
        best_own_distortion=_float_result(best_result.own_distortion),
        best_marginal_d2=_float_result(best_result.marginal_d2),
        best_soft_k_eff=_float_result(best_result.soft_k_eff),
        best_assignment_confidence=_float_result(best_result.assignment_confidence),
        best_assignment_entropy=_float_result(best_result.assignment_entropy),
        final_loss=final_loss,
        final_own_distortion=_float_result(final_result.own_distortion),
        final_marginal_d2=_float_result(final_result.marginal_d2),
        final_soft_k_eff=_float_result(final_result.soft_k_eff),
        assignment_confidence=_float_result(final_result.assignment_confidence),
        assignment_entropy=_float_result(final_result.assignment_entropy),
        finite=math.isfinite(best_loss),
        encountered_nonfinite=bool(encountered_nonfinite),
    )


def _optimize_restart(
    initial_logits: Tensor,
    geometry: SourceGeometry,
    *,
    restart_index: int,
    init_kind: str,
    seed: int,
    criterion: str,
    lambda_org: float,
    temperature: float,
    learning_rate: float,
    optimization_steps: int,
    device: torch.device,
) -> tuple[Tensor, RestartSummary]:
    """Optimize one initialized partition assignment."""
    logits = initial_logits.detach().to(device=device).requires_grad_(True)
    optimizer = torch.optim.Adam([logits], lr=float(learning_rate))
    encountered_nonfinite = False
    best_logits: Tensor | None = None
    best_result: SoftObjectiveResult | None = None
    best_loss = math.inf
    best_step = -1

    with torch.no_grad():
        initial_result = soft_objective(
            logits=logits,
            geometry=geometry,
            criterion=criterion,
            lambda_org=float(lambda_org),
            temperature=float(temperature),
        )
    if torch.isfinite(initial_result.loss):
        best_logits = logits.detach().cpu().clone()
        best_result = initial_result
        best_loss = _float_result(initial_result.loss)
        best_step = 0
    else:
        encountered_nonfinite = True

    final_result: SoftObjectiveResult = initial_result
    for step in range(int(optimization_steps)):
        optimizer.zero_grad(set_to_none=True)
        result = soft_objective(
            logits=logits,
            geometry=geometry,
            criterion=criterion,
            lambda_org=float(lambda_org),
            temperature=float(temperature),
        )
        if not torch.isfinite(result.loss):
            final_result = result
            encountered_nonfinite = True
            break
        result.loss.backward()
        if logits.grad is None or not torch.isfinite(logits.grad).all():
            final_result = result
            encountered_nonfinite = True
            break
        optimizer.step()

        with torch.no_grad():
            final_result = soft_objective(
                logits=logits,
                geometry=geometry,
                criterion=criterion,
                lambda_org=float(lambda_org),
                temperature=float(temperature),
            )
        if torch.isfinite(final_result.loss):
            current_loss = _float_result(final_result.loss)
            if current_loss < best_loss:
                best_loss = current_loss
                best_logits = logits.detach().cpu().clone()
                best_result = final_result
                best_step = step + 1
        else:
            encountered_nonfinite = True
            break

    with torch.no_grad():
        final_result = soft_objective(
            logits=logits,
            geometry=geometry,
            criterion=criterion,
            lambda_org=float(lambda_org),
            temperature=float(temperature),
        )
    if best_logits is None or best_result is None:
        best_logits = logits.detach().cpu()
        best_result = final_result
        best_step = -1
    summary = _restart_summary(
        restart_index=restart_index,
        init_kind=init_kind,
        seed=seed,
        best_result=best_result,
        best_step=best_step,
        final_result=final_result,
        encountered_nonfinite=encountered_nonfinite,
    )
    return best_logits.detach().cpu(), summary


def optimize_partition(
    geometry: SourceGeometry,
    *,
    criterion: str,
    lambda_org: float,
    num_partitions: int,
    temperature: float,
    learning_rate: float,
    optimization_steps: int,
    init_scale: float,
    restart_seeds: tuple[int, ...],
    include_collapse_restart: bool = False,
    collapse_init_bias: float = 4.0,
    initial_logit_specs: tuple[InitialLogitSpec, ...] = (),
    device: torch.device | str,
) -> OptimizedPartition:
    """Optimize a partition over the requested restart set."""
    if optimization_steps <= 0:
        raise ValueError("optimization_steps must be positive")
    if init_scale <= 0.0:
        raise ValueError("init_scale must be positive")
    if not restart_seeds and not initial_logit_specs:
        raise ValueError("At least one restart seed or initial logit specification is required")

    device = torch.device(device)
    device_geometry = geometry.to(device)
    best_loss = math.inf
    best_logits: Tensor | None = None
    best_restart = -1
    best_seed = -1
    summaries: list[RestartSummary] = []

    restart_specs: list[tuple[str, int, float | None]] = [("random", int(seed), None) for seed in restart_seeds]
    if include_collapse_restart and restart_seeds:
        restart_specs.insert(0, ("collapse", int(restart_seeds[0]), float(collapse_init_bias)))

    for restart_index, (init_kind, seed, collapse_bias) in enumerate(restart_specs):
        initial_logits = _seeded_logits(
            num_nodes=geometry.num_nodes,
            num_partitions=int(num_partitions),
            seed=int(seed),
            init_scale=float(init_scale),
            device=torch.device("cpu"),
            collapse_bias=collapse_bias,
        )
        final_logits, summary = _optimize_restart(
            initial_logits=initial_logits,
            geometry=device_geometry,
            restart_index=restart_index,
            init_kind=init_kind,
            seed=int(seed),
            criterion=criterion,
            lambda_org=lambda_org,
            temperature=temperature,
            learning_rate=learning_rate,
            optimization_steps=optimization_steps,
            device=device,
        )
        summaries.append(summary)
        if summary.finite and math.isfinite(summary.best_loss) and summary.best_loss < best_loss:
            best_loss = summary.best_loss
            best_logits = final_logits
            best_restart = restart_index
            best_seed = int(seed)

    for initial in initial_logit_specs:
        restart_index = len(summaries)
        final_logits, summary = _optimize_restart(
            initial_logits=initial.logits,
            geometry=device_geometry,
            restart_index=restart_index,
            init_kind=initial.init_kind,
            seed=int(initial.seed),
            criterion=criterion,
            lambda_org=lambda_org,
            temperature=temperature,
            learning_rate=learning_rate,
            optimization_steps=optimization_steps,
            device=device,
        )
        summaries.append(summary)
        if summary.finite and math.isfinite(summary.best_loss) and summary.best_loss < best_loss:
            best_loss = summary.best_loss
            best_logits = final_logits
            best_restart = restart_index
            best_seed = int(initial.seed)

    if best_logits is None or not math.isfinite(best_loss):
        raise RuntimeError(f"All restarts failed for criterion={criterion} lambda_org={lambda_org}")

    with torch.no_grad():
        selected_logits = best_logits.to(device=device)
        selected_soft = soft_objective(
            logits=selected_logits,
            geometry=device_geometry,
            criterion=criterion,
            lambda_org=float(lambda_org),
            temperature=float(temperature),
        )
        probabilities = torch.softmax(selected_logits / float(temperature), dim=-1)
        assignments = probabilities.argmax(dim=-1)

    return OptimizedPartition(
        criterion=criterion,
        lambda_org=float(lambda_org),
        selected_restart=int(best_restart),
        selected_seed=int(best_seed),
        logits=best_logits,
        probabilities=probabilities.detach().cpu(),
        assignments=assignments.detach().cpu(),
        soft_result=selected_soft,
        restart_summaries=summaries,
    )
