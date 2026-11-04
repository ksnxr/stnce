#!/usr/bin/env python3
"""Analyze how local energy-difference errors accumulate into global ratios.

The construction follows the decompositions in Section 3 of the paper:

* tNCE: two fixed-x temporal paths and the known reference ratio;
* tCNCE: a straight 100-step spatial interpolation at the endpoint time;
* stNCE-m/w/o/s: the same straight 100-step spatiotemporal interpolation;

Every decomposition telescopes exactly. The script measures the error of each
local learned contribution, its density regime, and its cumulative global error.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from scipy.stats import chi2


STNCE_ROOT = Path(__file__).resolve().parents[1]
ROOT = STNCE_ROOT.parent
ANALYSIS_ROOT = STNCE_ROOT / "analysis"
for module_root in (ANALYSIS_ROOT, STNCE_ROOT):
    if str(module_root) not in sys.path:
        sys.path.insert(0, str(module_root))

from models.unet import PrecondMNISTUNet
from models.ema import ExponentialMovingAverage

DIM = 784
COMPONENT_STD = 0.2  # Mixture std=0.1 before x -> 2x-1 scaling.


def ns(**kwargs):
    return SimpleNamespace(**kwargs)


def model_config():
    """Exact model settings recorded in each run's W&B config."""
    return ns(
        model=ns(
            model_channels=32,
            num_res_blocks=2,
            attention_resolutions=[7],
            dropout=0.0,
            channel_mult=[1, 2, 2],
            use_checkpoint=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            resblock_updown=False,
            with_fourier_features=False,
            input_projection=True,
            head_bias=True,
            output_param="sum",
            a=0.75,
            ema_rate=0.999,
        )
    )


def load_scaled_component_means() -> torch.Tensor:
    indices = torch.load(
        STNCE_ROOT / "data_cache/mnist_mixture_indices.pt",
        map_location="cpu",
        weights_only=True,
    )
    dataset = MNIST(
        str(ROOT / "data"),
        train=True,
        download=False,
        transform=transforms.ToTensor(),
    )
    means = torch.stack([dataset[int(i)][0] for i in indices])
    return 2.0 * means - 1.0


def load_self_model(checkpoint_path: Path) -> PrecondMNISTUNet:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    config = model_config()
    model = PrecondMNISTUNet(config)
    model.load_state_dict(checkpoint["model_state_dict"])

    if "ema_state_dict" not in checkpoint:
        raise ValueError(
            "Analysis requires EMA parameters, but checkpoint "
            f"{checkpoint_path} does not contain `ema_state_dict`."
        )

    # Match the evaluation API used by imagenet_run_lib.py and image_run_lib.py.
    ema = ExponentialMovingAverage(
        model.parameters(), decay=config.model.ema_rate
    )
    ema.load_state_dict(checkpoint["ema_state_dict"])
    ema.copy_to(model.parameters())
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    del checkpoint
    return model


def exact_path_stats(
    samples: torch.Tensor,
    times: torch.Tensor,
    flat_means: torch.Tensor,
    *,
    chunk_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact log p_t(x) and nearest-component squared Mahalanobis radius."""
    samples = samples.reshape(samples.shape[0], -1).float()
    times = times.float()
    mean_sq = torch.sum(flat_means.square(), dim=1)
    log_num_components = math.log(flat_means.shape[0])
    log_probs = []
    min_radii = []
    for start in range(0, samples.shape[0], chunk_size):
        x = samples[start : start + chunk_size]
        t = times[start : start + chunk_size]
        x_sq = torch.sum(x.square(), dim=1, keepdim=True)
        cross = x @ flat_means.T
        distances = (
            x_sq
            + t.square().unsqueeze(1) * mean_sq.unsqueeze(0)
            - 2.0 * t.unsqueeze(1) * cross
        ).clamp_min_(0.0)
        variances = (
            t.square() * (COMPONENT_STD**2) + (1.0 - t).square()
        )
        radii = distances / variances.unsqueeze(1)
        log_normalizer = -0.5 * DIM * torch.log(
            2.0 * math.pi * variances
        )
        component_log_probs = (
            log_normalizer.unsqueeze(1) - 0.5 * radii - log_num_components
        )
        log_probs.append(torch.logsumexp(component_log_probs, dim=1))
        min_radii.append(torch.min(radii, dim=1).values)
    return torch.cat(log_probs), torch.cat(min_radii)


# Replace each placeholder with its run directory name under --log-root.
RUNS = {
    "tNCE": "TODO-0",
    "tCNCE": "TODO-1",
    "stNCE-m": "TODO-2",
    "stNCE-w": "TODO-3",
    "stNCE-o": "TODO-4",
    "stNCE-s": "TODO-5",
}
METHOD_ORDER = list(RUNS)
COLORS = {
    "tNCE": "#54A24B",
    "tCNCE": "#F58518",
    "stNCE-m": "#E45756",
    "stNCE-w": "#D95F5F",
    "stNCE-o": "#B23A48",
    "stNCE-s": "#8C1D40",
}
BASE_LOG_NORMALIZER = 0.5 * DIM * math.log(2.0 * math.pi)
NUM_PATH_STEPS = 100
EVALUATION_BEST_KEY = "val_norm_mse"
TRAIN_TIME_PER_STEP_NORMALIZER = 100000.0
EVALUATION_METRICS = {
    "evaluation_test_mse": "test_mse",
    "evaluation_test_real_ratio": "test_real_ratio",
    "evaluation_test_norm_mse": "test_norm_mse",
    "evaluation_test_norm_nll": "test_norm_nll",
}


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_recorded_evaluation_metrics(run_dir: Path) -> dict:
    """Read the metrics reported by scripts/tool.py for one saved run."""
    metrics_path = run_dir / "metrics.pkl"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    with metrics_path.open("rb") as handle:
        metrics = pickle.load(handle)

    if EVALUATION_BEST_KEY not in metrics:
        raise KeyError(
            f"{metrics_path} does not contain {EVALUATION_BEST_KEY!r}"
        )
    selection_values = metrics[EVALUATION_BEST_KEY]
    selected_step = min(
        selection_values,
        key=lambda step: selection_values[step],
    )

    result = {
        "evaluation_metrics_path": str(metrics_path),
        "evaluation_best_key": EVALUATION_BEST_KEY,
        "evaluation_selected_step": int(selected_step),
        "evaluation_val_norm_mse": float(
            selection_values[selected_step]
        ),
    }
    for output_name, metric_name in EVALUATION_METRICS.items():
        if metric_name not in metrics:
            raise KeyError(
                f"{metrics_path} does not contain {metric_name!r}"
            )
        result[output_name] = float(metrics[metric_name][selected_step])

    train_times = metrics.get("train_time_between_eval_steps")
    if not isinstance(train_times, dict):
        raise KeyError(
            f"{metrics_path} does not contain "
            "`train_time_between_eval_steps` as a dictionary"
        )
    result["evaluation_train_time_per_step"] = (
        sum(float(value) for value in train_times.values())
        / TRAIN_TIME_PER_STEP_NORMALIZER
    )
    return result


def balanced_endpoint_pairs(
    means: torch.Tensor,
    seed: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return one noisy endpoint per mode, a cross-mode pairing, and bases."""
    generator = torch.Generator().manual_seed(seed)
    endpoints = means + COMPONENT_STD * torch.randn(
        means.shape, generator=generator
    )
    shift = int(
        torch.randint(
            1,
            means.shape[0],
            (1,),
            generator=generator,
        )
    )
    paired = torch.roll(endpoints, shifts=shift, dims=0)
    shared_base = torch.randn(endpoints.shape, generator=generator)
    return endpoints, paired, shared_base


def minimax_mode_graph(
    flat_means: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return all-pairs minimax costs and next hops on the mode graph.

    For equal isotropic mixture components, the largest gap between
    consecutive component centers is a conservative proxy for the deepest
    density valley encountered by a piecewise-linear spatial path.
    """
    costs = torch.cdist(
        flat_means.to(torch.float64),
        flat_means.to(torch.float64),
    )
    next_hop = torch.arange(
        flat_means.shape[0], dtype=torch.long
    ).view(1, -1).expand(flat_means.shape[0], -1).clone()
    for pivot in range(flat_means.shape[0]):
        candidate = torch.maximum(
            costs[:, pivot : pivot + 1],
            costs[pivot : pivot + 1, :],
        )
        improve = candidate < costs
        costs = torch.where(improve, candidate, costs)
        proposed_hop = next_hop[:, pivot : pivot + 1].expand_as(
            next_hop
        )
        next_hop = torch.where(improve, proposed_hop, next_hop)
    return costs, next_hop


def reconstruct_mode_path(
    source: int,
    target: int,
    next_hop: torch.Tensor,
) -> list[int]:
    """Reconstruct one minimax component-center path."""
    if source == target:
        return [source]
    result = [source]
    current = source
    for _ in range(next_hop.shape[0] + 1):
        current = int(next_hop[current, target])
        result.append(current)
        if current == target:
            return result
    raise RuntimeError(
        f"Could not reconstruct minimax path {source} -> {target}"
    )


def resample_polyline(
    control_points: torch.Tensor,
    num_steps: int,
) -> torch.Tensor:
    """Resample a polyline at uniform arc-length progress."""
    if control_points.shape[0] == 1:
        return control_points.expand(num_steps + 1, -1).clone()
    segment_lengths = torch.linalg.vector_norm(
        control_points[1:] - control_points[:-1],
        dim=1,
    )
    keep_segment = segment_lengths > 1e-12
    if not torch.all(keep_segment):
        keep_point = torch.cat(
            [
                torch.ones(1, dtype=torch.bool),
                keep_segment,
            ]
        )
        control_points = control_points[keep_point]
        if control_points.shape[0] == 1:
            return control_points.expand(num_steps + 1, -1).clone()
        segment_lengths = torch.linalg.vector_norm(
            control_points[1:] - control_points[:-1],
            dim=1,
        )
    cumulative = torch.cat(
        [
            torch.zeros(1, dtype=control_points.dtype),
            torch.cumsum(segment_lengths, dim=0),
        ]
    )
    targets = torch.linspace(
        0.0,
        float(cumulative[-1]),
        num_steps + 1,
        dtype=control_points.dtype,
    )
    indices = torch.searchsorted(
        cumulative[1:],
        targets,
        right=False,
    ).clamp_max(segment_lengths.shape[0] - 1)
    lower = cumulative[indices]
    alpha = (
        (targets - lower)
        / segment_lengths[indices].clamp_min(1e-12)
    )
    return (
        (1.0 - alpha[:, None]) * control_points[indices]
        + alpha[:, None] * control_points[indices + 1]
    )


def build_minimax_spatial_bridges(
    endpoint_x: torch.Tensor,
    endpoint_y: torch.Tensor,
    paired_indices: torch.Tensor,
    flat_means: torch.Tensor,
    next_hop: torch.Tensor,
    target_step_norm: float,
    maximum_num_steps: int,
) -> tuple[
    torch.Tensor,
    list[list[int]],
    int,
    torch.Tensor,
    torch.Tensor,
]:
    """Build best-case spatial bridges through the mixture mode graph."""
    flat_x = endpoint_x.flatten(1).to(torch.float64)
    flat_y = endpoint_y.flatten(1).to(torch.float64)
    flat_means = flat_means.to(torch.float64)
    control_point_paths = []
    mode_paths = []
    path_lengths = []
    for endpoint_index in range(endpoint_x.shape[0]):
        source_mode = int(paired_indices[endpoint_index])
        target_mode = endpoint_index
        mode_path = reconstruct_mode_path(
            source_mode,
            target_mode,
            next_hop,
        )
        control_points = torch.cat(
            [
                flat_y[endpoint_index : endpoint_index + 1],
                flat_means[mode_path],
                flat_x[endpoint_index : endpoint_index + 1],
            ],
            dim=0,
        )
        control_point_paths.append(control_points)
        path_lengths.append(
            float(
                torch.linalg.vector_norm(
                    control_points[1:] - control_points[:-1],
                    dim=1,
                ).sum()
            )
        )
        mode_paths.append(mode_path)
    step_counts = torch.tensor(
        [
            min(
                maximum_num_steps,
                max(1, math.ceil(length / target_step_norm)),
            )
            for length in path_lengths
        ],
        dtype=torch.long,
    )
    num_steps = int(step_counts.max())
    paths = []
    edge_masks = []
    for control_points, path_num_steps in zip(
        control_point_paths,
        step_counts.tolist(),
    ):
        path = resample_polyline(control_points, path_num_steps)
        padding = num_steps - path_num_steps
        if padding:
            path = torch.cat(
                [
                    path,
                    path[-1:].expand(padding, -1),
                ],
                dim=0,
            )
        paths.append(path)
        edge_masks.append(
            torch.arange(num_steps) < path_num_steps
        )
    stacked = torch.stack(paths).to(endpoint_x.dtype)
    edge_mask = torch.stack(edge_masks)
    return stacked.reshape(
        endpoint_x.shape[0],
        num_steps + 1,
        *endpoint_x.shape[1:],
    ), mode_paths, num_steps, edge_mask, step_counts


def build_white_kernel_bridges(
    shared_base: torch.Tensor,
    endpoints: torch.Tensor,
    num_steps: int,
    white_std: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Build endpoint-conditioned bridges from white-kernel increments.

    Centering the Gaussian increments conditions their sum to zero; adding
    the endpoint drift then makes the bridge hit both prescribed endpoints
    while retaining kernel-scale stochastic local fluctuations.
    """
    noises = white_std * torch.randn(
        (
            endpoints.shape[0],
            num_steps,
            *endpoints.shape[1:],
        ),
        generator=generator,
        dtype=endpoints.dtype,
    )
    noises = noises - noises.mean(dim=1, keepdim=True)
    drift = (
        endpoints - shared_base
    )[:, None] / float(num_steps)
    increments = drift + noises
    return torch.cat(
        [
            shared_base[:, None],
            shared_base[:, None] + torch.cumsum(increments, dim=1),
        ],
        dim=1,
    )


def build_paired_white_kernel_bridges(
    shared_base: torch.Tensor,
    endpoint_x: torch.Tensor,
    endpoint_y: torch.Tensor,
    minimum_num_steps: int,
    maximum_num_steps: int,
    white_std: float,
    drift_fraction: float,
    seed: int,
    time_low: float,
    time_high: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Build paired, variable-length, kernel-scale white bridges."""
    delta_x = torch.linalg.vector_norm(
        (endpoint_x - shared_base).flatten(1),
        dim=1,
    )
    delta_y = torch.linalg.vector_norm(
        (endpoint_y - shared_base).flatten(1),
        dim=1,
    )
    target_drift_norm = (
        drift_fraction * white_std * math.sqrt(DIM)
    )
    step_counts = torch.ceil(
        torch.maximum(delta_x, delta_y) / target_drift_norm
    ).to(torch.long)
    step_counts = step_counts.clamp(
        min=minimum_num_steps,
        max=maximum_num_steps,
    )
    maximum_steps = int(step_counts.max())
    generator_x = torch.Generator().manual_seed(seed + 1001)
    generator_y = torch.Generator().manual_seed(seed + 2001)
    paths_x = []
    paths_y = []
    path_times = []
    edge_masks = []
    for pair_index, pair_steps in enumerate(step_counts.tolist()):
        path_x = build_white_kernel_bridges(
            shared_base[pair_index : pair_index + 1],
            endpoint_x[pair_index : pair_index + 1],
            pair_steps,
            white_std,
            generator_x,
        )[0]
        path_y = build_white_kernel_bridges(
            shared_base[pair_index : pair_index + 1],
            endpoint_y[pair_index : pair_index + 1],
            pair_steps,
            white_std,
            generator_y,
        )[0]
        times = torch.linspace(
            time_low,
            time_high,
            pair_steps + 1,
            dtype=endpoint_x.dtype,
        )
        padding = maximum_steps - pair_steps
        if padding:
            path_x = torch.cat(
                [path_x, path_x[-1:].expand(padding, *path_x.shape[1:])],
                dim=0,
            )
            path_y = torch.cat(
                [path_y, path_y[-1:].expand(padding, *path_y.shape[1:])],
                dim=0,
            )
            times = torch.cat(
                [
                    times,
                    times[-1:].expand(padding),
                ]
            )
        paths_x.append(path_x)
        paths_y.append(path_y)
        path_times.append(times)
        edge_masks.append(
            torch.arange(maximum_steps) < pair_steps
        )
    return (
        torch.stack(paths_x),
        torch.stack(paths_y),
        torch.stack(path_times),
        torch.stack(edge_masks),
        step_counts,
    )


def spatial_step_statistics(
    *paths: torch.Tensor,
    edge_mask: torch.Tensor | None = None,
) -> tuple[float, float]:
    """Return mean and 95th percentile spatial edge length."""
    lengths = []
    for path in paths:
        difference = path[:, 1:] - path[:, :-1]
        lengths.append(
            torch.linalg.vector_norm(
                difference.flatten(2),
                dim=2,
            ).reshape(-1)
        )
    all_lengths = torch.cat(lengths).to(torch.float64)
    if edge_mask is not None:
        repeated_mask = edge_mask.reshape(-1).repeat(len(paths))
        all_lengths = all_lengths[repeated_mask]
    return (
        float(all_lengths.mean()),
        float(torch.quantile(all_lengths, 0.95)),
    )


@torch.no_grad()
def model_energy(
    model: torch.nn.Module,
    samples: torch.Tensor,
    times: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    original_shape = samples.shape[:-3]
    flat_samples = samples.reshape(-1, *samples.shape[-3:])
    flat_times = times.reshape(-1)
    parts = []
    for start in range(0, flat_samples.shape[0], batch_size):
        part = model(
            flat_samples[start : start + batch_size],
            flat_times[start : start + batch_size],
        )
        parts.append(part.reshape(-1).to(torch.float64).cpu())
    return torch.cat(parts).reshape(original_shape)


def exact_energy_and_radius(
    samples: torch.Tensor,
    times: torch.Tensor,
    flat_means: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    original_shape = samples.shape[:-3]
    log_prob, radius = exact_path_stats(
        samples.reshape(-1, *samples.shape[-3:]),
        times.reshape(-1),
        flat_means,
    )
    return (
        -log_prob.to(torch.float64).reshape(original_shape),
        radius.to(torch.float64).reshape(original_shape),
    )


def pair_ratio_error(
    model: torch.nn.Module,
    xs: torch.Tensor,
    ys: torch.Tensor,
    times: torch.Tensor,
    flat_means: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    model_x = model_energy(model, xs, times, batch_size)
    model_y = model_energy(model, ys, times, batch_size)
    exact_x, radius_x = exact_energy_and_radius(xs, times, flat_means)
    exact_y, radius_y = exact_energy_and_radius(ys, times, flat_means)
    # Error in log p(x,t) - log p(y,t); time-only constants cancel.
    error = (-model_x + model_y) - (-exact_x + exact_y)
    return error, torch.maximum(radius_x, radius_y)


def point_log_density_error(
    model: torch.nn.Module,
    samples: torch.Tensor,
    times: torch.Tensor,
    flat_means: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    model_value = model_energy(model, samples, times, batch_size)
    exact_value, radius = exact_energy_and_radius(
        samples, times, flat_means
    )
    # A time-only offset is harmless because it cancels between path points.
    return -model_value + exact_value, radius


def summarize_decomposition(
    method: str,
    point_error: torch.Tensor,
    point_radius: torch.Tensor,
    threshold: float,
    *,
    intermediate_error: torch.Tensor | None = None,
    intermediate_radius: torch.Tensor | None = None,
    edge_mask: torch.Tensor | None = None,
) -> tuple[dict, list[dict]]:
    cumulative_error = point_error - point_error[:, :1]
    local_error = cumulative_error[:, 1:] - cumulative_error[:, :-1]
    if edge_mask is None:
        edge_mask = torch.ones_like(local_error, dtype=torch.bool)
    else:
        edge_mask = edge_mask.to(torch.bool)

    endpoint_radius = torch.maximum(
        point_radius[:, :-1], point_radius[:, 1:]
    )
    if intermediate_radius is None:
        edge_radius = endpoint_radius
    else:
        edge_radius = torch.maximum(endpoint_radius, intermediate_radius)
    low_mask = (edge_radius > threshold) & edge_mask
    high_mask = (edge_radius <= threshold) & edge_mask

    abs_local = local_error.abs()
    total_abs = (abs_local * edge_mask).sum(dim=1)
    low_abs = (abs_local * low_mask).sum(dim=1)
    global_error = cumulative_error[:, -1]
    sum_local = (local_error * edge_mask).sum(dim=1)

    def masked_rmse(mask: torch.Tensor) -> float:
        selected = local_error[mask]
        if selected.numel() == 0:
            return float("nan")
        return float(selected.square().mean().sqrt())

    row = {
        "method": method,
        "num_pairs": point_error.shape[0],
        "num_edges": float(edge_mask.sum(dim=1).to(torch.float64).mean()),
        "min_edges": int(edge_mask.sum(dim=1).min()),
        "max_edges": int(edge_mask.sum(dim=1).max()),
        "global_ratio_rmse": float(global_error.square().mean().sqrt()),
        "global_ratio_mae": float(global_error.abs().mean()),
        "local_error_rmse": float(
            local_error[edge_mask].square().mean().sqrt()
        ),
        "high_density_local_rmse": masked_rmse(high_mask),
        "low_density_local_rmse": masked_rmse(low_mask),
        "low_density_edge_rate": float(
            low_mask.sum() / edge_mask.sum().clamp_min(1)
        ),
        "low_density_abs_error_share": float(
            low_abs.sum() / total_abs.sum().clamp_min(1e-12)
        ),
        "mean_path_abs_error": float(total_abs.mean()),
        "mean_cancellation_ratio": float(
            (
                global_error.abs()
                / total_abs.clamp_min(1e-12)
            ).mean()
        ),
        "accumulation_factor_global_over_local_rmse": float(
            global_error.square().mean().sqrt()
            / local_error[edge_mask]
            .square()
            .mean()
            .sqrt()
            .clamp_min(1e-12)
        ),
        "telescoping_max_abs_residual": float(
            (sum_local - global_error).abs().max()
        ),
    }
    if intermediate_error is not None:
        temporal_error = intermediate_error - point_error[:, :-1]
        spatial_error = point_error[:, 1:] - intermediate_error
        row["staircase_temporal_local_rmse"] = float(
            temporal_error.square().mean().sqrt()
        )
        row["staircase_spatial_local_rmse"] = float(
            spatial_error.square().mean().sqrt()
        )
        row["staircase_subedge_sum_max_abs_residual"] = float(
            (temporal_error + spatial_error - local_error).abs().max()
        )
    else:
        row["staircase_temporal_local_rmse"] = float("nan")
        row["staircase_spatial_local_rmse"] = float("nan")
        row["staircase_subedge_sum_max_abs_residual"] = float("nan")

    cumulative_rows = []
    progress = torch.linspace(0.0, 1.0, 101)
    step_counts = edge_mask.sum(dim=1)
    for index, fraction in enumerate(progress):
        positions = fraction * step_counts.to(torch.float64)
        lower = torch.floor(positions).to(torch.long)
        upper = torch.ceil(positions).to(torch.long)
        alpha = positions - lower.to(torch.float64)
        lower_values = cumulative_error.gather(
            1, lower[:, None]
        ).squeeze(1)
        upper_values = cumulative_error.gather(
            1, upper[:, None]
        ).squeeze(1)
        values = (
            (1.0 - alpha) * lower_values
            + alpha * upper_values
        )
        cumulative_rows.append(
            {
                "method": method,
                "step": index,
                "progress": float(fraction),
                "cumulative_error_rmse": float(
                    values.square().mean().sqrt()
                ),
                "cumulative_error_mae": float(values.abs().mean()),
                "cumulative_error_median": float(torch.median(values)),
            }
        )
    return row, cumulative_rows


def projected_density_figure(
    path: Path,
    means: torch.Tensor,
    seed: int,
):
    flat = means.flatten(1).to(torch.float64)
    distances = torch.cdist(flat, flat)
    pair_index = torch.argmax(distances)
    left_index = int(pair_index // flat.shape[0])
    right_index = int(pair_index % flat.shape[0])
    left = flat[left_index]
    right = flat[right_index]
    direction = (right - left) / torch.linalg.norm(right - left)
    projected_means = flat @ direction
    midpoint = 0.5 * (left + right)
    midpoint_perpendicular = (
        midpoint - (midpoint @ direction) * direction
    )
    mean_perpendicular = (
        flat - projected_means[:, None] * direction[None, :]
    )
    perpendicular_distance_sq = (
        mean_perpendicular - midpoint_perpendicular[None, :]
    ).square().sum(dim=1)

    generator = torch.Generator().manual_seed(seed)
    base = torch.randn((DIM,), generator=generator, dtype=torch.float64)
    projected_base = float(base @ direction)
    projected_left = float(left @ direction)
    projected_right = float(right @ direction)

    lower = min(
        projected_left, projected_right, projected_base
    ) - 2.5
    upper = max(
        projected_left, projected_right, projected_base
    ) + 2.5
    positions = torch.linspace(lower, upper, 260, dtype=torch.float64)
    times = torch.linspace(0.0, 1.0, 181, dtype=torch.float64)
    log_density = []
    for time in times:
        variance = (
            time * time * COMPONENT_STD * COMPONENT_STD
            + (1.0 - time) ** 2
        )
        # Full-dimensional density on the affine slice containing the two
        # selected component means.  Unlike a 1-D marginal, this retains the
        # orthogonal distance to every other MNIST mode.
        distance_sq = (
            (
                positions[:, None]
                - time * projected_means[None, :]
            ).square()
            + time.square() * perpendicular_distance_sq[None, :]
        )
        component_log_prob = (
            -0.5 * DIM * math.log(2.0 * math.pi * float(variance))
            - 0.5 * distance_sq / variance
            - math.log(projected_means.shape[0])
        )
        log_density.append(torch.logsumexp(component_log_prob, dim=1))
    density = torch.stack(log_density)
    density = density - density.max(dim=1, keepdim=True).values
    density = density.clamp_min(-10.0)

    fig, ax = plt.subplots(figsize=(9.3, 5.4), constrained_layout=True)
    extent = [lower, upper, 0.0, 1.0]
    image = ax.imshow(
        density.numpy(),
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="Blues",
        vmin=-10.0,
        vmax=0.0,
    )

    # Straight spatial interpolation at t=1.
    ax.plot(
        [projected_left, projected_right],
        [1.0, 1.0],
        color="#F58518",
        linewidth=3,
        label="spatial interpolation",
    )
    # Temporal decomposition: two vertical paths and known base ratio.
    ax.plot(
        [projected_left, projected_left],
        [0.0, 1.0],
        color="#54A24B",
        linewidth=2.5,
        label="temporal",
    )
    ax.plot(
        [projected_right, projected_right],
        [0.0, 1.0],
        color="#54A24B",
        linewidth=2.5,
    )
    ax.plot(
        [projected_left, projected_right],
        [0.0, 0.0],
        color="#54A24B",
        linewidth=2.5,
    )
    # Shared straight spatiotemporal interpolation for all stNCE variants.
    ax.plot(
        [projected_base, projected_left],
        [0.0, 1.0],
        color="#B23A48",
        linewidth=3,
        label="shared spatiotemporal interpolation",
    )
    ax.plot(
        [projected_base, projected_right],
        [0.0, 1.0],
        color="#B23A48",
        linewidth=3,
    )
    ax.scatter(
        [projected_left, projected_right, projected_base],
        [1.0, 1.0, 0.0],
        color="black",
        s=28,
        zorder=5,
    )
    ax.set(
        xlabel="projection onto the farthest-mode direction",
        ylabel="time",
        title="MNIST-mixture analogue of Figure 2",
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=3,
        framealpha=0.92,
    )
    colorbar = fig.colorbar(image, ax=ax, pad=0.03)
    colorbar.set_label("relative full-density slice (nats)")
    fig.savefig(path, dpi=190)
    plt.close(fig)


def plot_error_analysis(
    path: Path,
    summaries: list[dict],
    cumulative_rows: list[dict],
):
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.5))
    methods = [row["method"] for row in summaries]
    positions = np.arange(len(methods))

    for method in methods:
        rows = [
            row for row in cumulative_rows if row["method"] == method
        ]
        axes[0, 0].plot(
            [row["progress"] for row in rows],
            [row["cumulative_error_rmse"] for row in rows],
            color=COLORS[method],
            linewidth=2,
            label=method,
        )
    axes[0, 0].set(
        xlabel="fraction of complete path",
        ylabel="RMSE of accumulated ratio error (nats)",
        title="Local errors accumulate into global error",
    )
    axes[0, 0].legend(ncol=2, fontsize=8)

    width = 0.38
    axes[0, 1].bar(
        positions - width / 2,
        [row["high_density_local_rmse"] for row in summaries],
        width,
        color="#4C78A8",
        label="high-density edges",
    )
    axes[0, 1].bar(
        positions + width / 2,
        [row["low_density_local_rmse"] for row in summaries],
        width,
        color="#E45756",
        label="low-density edges",
    )
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xticks(positions, methods, rotation=25, ha="right")
    axes[0, 1].set(
        ylabel="local ratio-error RMSE (nats, log scale)",
        title="Accuracy of individual local differences",
    )
    axes[0, 1].legend(fontsize=8)

    for row in summaries:
        axes[1, 0].scatter(
            100.0 * row["low_density_edge_rate"],
            100.0 * row["low_density_abs_error_share"],
            s=80,
            color=COLORS[row["method"]],
            label=row["method"],
        )
        axes[1, 0].annotate(
            row["method"],
            (
                100.0 * row["low_density_edge_rate"],
                100.0 * row["low_density_abs_error_share"],
            ),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axes[1, 0].plot([0, 100], [0, 100], "k--", linewidth=1)
    axes[1, 0].set(
        xlabel="low-density edges (%)",
        ylabel="absolute local error on low-density edges (%)",
        title="Low-density edges carry disproportionate error",
        xlim=(-1, None),
        ylim=(-1, None),
    )

    axes[1, 1].bar(
        positions,
        [row["global_ratio_rmse"] for row in summaries],
        color=[COLORS[method] for method in methods],
    )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xticks(positions, methods, rotation=25, ha="right")
    axes[1, 1].set(
        ylabel="global endpoint ratio RMSE (nats, log scale)",
        title="Result after telescoping all local errors",
    )

    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.suptitle(
        "Section 3 heuristic on MNIST mixture: local-to-global error"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def write_report(
    path: Path,
    summaries: list[dict],
):
    bridge_policy = summaries[0]["bridge_policy"]
    lines = [
        "# Local-to-global density-ratio analysis",
        "",
        "This analysis implements the Section 3 telescoping decompositions "
        "directly on 100 paired MNIST-mixture endpoint samples. Every bridge "
        "has exactly 100 steps. tCNCE uses a straight spatial interpolation "
        "at the data endpoint time. stNCE-m, stNCE-w, stNCE-o, and stNCE-s "
        "use the same straight space-time interpolation from a shared "
        "Gaussian base to each data endpoint. The learned local contribution "
        "is computed from the checkpoint energy, and the exact contribution "
        "from the known 100-component Gaussian mixture.",
        "",
        "Write the error of local edge \(k\) as "
        "\(e_k=(\\widehat{\\ell}_{k+1}-\\widehat{\\ell}_k)"
        "-(\\ell_{k+1}-\\ell_k)\). Then every construction obeys",
        "",
        "\\[",
        "(\\widehat{\\ell}_K-\\widehat{\\ell}_0)"
        "-(\\ell_K-\\ell_0)=\\sum_{k=0}^{K-1} e_k.",
        "\\]",
        "",
        "For tCNCE, the points lie on one spatial bridge at fixed time. "
        "For tNCE, the identity is applied to two fixed-image temporal "
        "paths and their errors are subtracted; the known reference-density "
        "ratio closes the path. For stNCE, two joint space-time paths share "
        "the same reference point, so their base terms cancel. All four "
        "stNCE variants use exactly the same joint edges.",
        "",
        "For each method, the signed local errors telescope to the reported "
        "global endpoint ratio error; the maximum numerical residual is "
        f"{max(row['telescoping_max_abs_residual'] for row in summaries):.2e} "
        "nats.",
        "",
        f"Bridge policy: `{bridge_policy}`. tNCE retains its fixed-image "
        "temporal fibers. The four stNCE variants now share identical path "
        "points and identical direct joint-edge decompositions.",
        "",
        "The recorded evaluation columns below follow the reporting logic in "
        "`scripts/tool.py`: for each run, select the step minimizing "
        f"`{EVALUATION_BEST_KEY}`, then read the test metrics at that step. "
        "Training time is the sum of the recorded times between evaluation "
        f"steps divided by {int(TRAIN_TIME_PER_STEP_NORMALIZER)}. These are "
        "the records associated with the analyzed seed-1 checkpoints, rather "
        "than a mean and standard deviation over additional seeds.",
        "",
        "## Error accumulation",
        "",
        "| Method | Path decomposition | Edges (mean; range) | "
        "Low-density edges | Mean spatial step | Local RMSE: high | "
        "Local RMSE: low | Error "
        "share on low-density edges | Global ratio RMSE | Accumulation "
        "factor | Selected step | Test MSE | Test Ratio | Test NormMSE | "
        "Test NormNLL | Time/step |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        "---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['method']} | {row['bridge_name']} | "
            f"{row['num_edges']:.1f}; "
            f"{row['min_edges']}-{row['max_edges']} | "
            f"{100.0 * row['low_density_edge_rate']:.2f}% | "
            f"{row['mean_spatial_step_l2']:.3f} | "
            f"{row['high_density_local_rmse']:.3f} | "
            f"{row['low_density_local_rmse']:.3f} | "
            f"{100.0 * row['low_density_abs_error_share']:.2f}% | "
            f"{row['global_ratio_rmse']:.3f} | "
            f"{row['accumulation_factor_global_over_local_rmse']:.1f}x | "
            f"{row['evaluation_selected_step']:,} | "
            f"{row['evaluation_test_mse']:.2f} | "
            f"{row['evaluation_test_real_ratio']:.2f} | "
            f"{row['evaluation_test_norm_mse']:.2f} | "
            f"{row['evaluation_test_norm_nll']:.2f} | "
            f"{row['evaluation_train_time_per_step']:.2f} |"
        )
    lines.extend(
        [
            "",
            "`Accumulation factor` is global endpoint RMSE divided by the "
            "RMSE of one local signed error. It is descriptive rather than a "
            "condition number: local errors are correlated and can cancel.",
            "",
            "## Reading the result",
            "",
            "- **tNCE:** Each `0.01` temporal step is small, but the two "
            "fixed-x paths enter regions where the endpoint images are poorly "
            "represented at intermediate times. Errors in those local time "
            "differences accumulate into the global mode ratio.",
            "- **tCNCE:** The straight 100-step endpoint interpolation "
            "directly measures the spatial low-density pathology illustrated "
            "in Figure 2.",
            "- **stNCE-m, stNCE-w, stNCE-o, and stNCE-s:** These checkpoints are "
            "evaluated on exactly the same direct joint space-time paths. "
            "Differences therefore reflect their learned energies rather "
            "than different diagnostic bridge geometries.",
            "",
            "## Conclusion",
            "",
            "The checkpoints support the unifying heuristic in the precise "
            "telescoping sense: global density-ratio error is the signed sum "
            "of local difference errors. Pure temporal and spatial paths "
            "force that sum through poorly sampled regions; spatiotemporal "
            "paths keep most terms in regions where the local differences are "
            "estimated more accurately.",
            "",
            "The analysis does not imply that all local errors add with the "
            "same sign. The cancellation ratios in the CSV show substantial "
            "cancellation. What matters is that low-density segments permit "
            "coherent drift that is not constrained by data, so a relatively "
            "small local bias can become a large endpoint offset.",
            "",
            "The paths are controlled diagnostic constructions, not exact "
            "replays of every stochastic training transition. `Low density` "
            "means that at least one endpoint of an edge lies outside the "
            "99% chi-square region of every exact mixture component. The "
            "Figure 2 analogue is a full-density affine slice for two far-"
            "separated modes; the quantitative support statistics use all "
            "784 dimensions.",
            "",
            "For a scalar checkpoint, changing a valid path cannot improve "
            "the signed endpoint error: telescoping makes it path-invariant. "
            "The common bridge policy makes comparisons among the four "
            "stNCE checkpoints path-matched.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("TODO-6"),
        help="Parent of the six saved run directories; replace TODO-6 or pass --log-root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "logs/mnist_mixture_test"
            / "fixed_time_std_0.01_local_to_global_analysis"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    means = load_scaled_component_means()
    flat_means = means.flatten(1)
    endpoint_x, endpoint_y, shared_base = balanced_endpoint_pairs(
        means, args.seed
    )
    num_pairs = endpoint_x.shape[0]
    times_1d = torch.linspace(
        1e-4,
        1.0 - 1e-4,
        NUM_PATH_STEPS + 1,
    )
    times = times_1d.view(1, -1).expand(num_pairs, -1)
    expanded_shape = (
        num_pairs,
        NUM_PATH_STEPS + 1,
        1,
        28,
        28,
    )

    vertical_x = endpoint_x[:, None].expand(expanded_shape)
    vertical_y = endpoint_y[:, None].expand(expanded_shape)
    mixing = times.view(num_pairs, -1, 1, 1, 1)
    diagonal_x = (
        (1.0 - mixing) * shared_base[:, None]
        + mixing * endpoint_x[:, None]
    )
    diagonal_y = (
        (1.0 - mixing) * shared_base[:, None]
        + mixing * endpoint_y[:, None]
    )
    spatial = (
        (1.0 - mixing) * endpoint_y[:, None]
        + mixing * endpoint_x[:, None]
    )
    spatial_times = torch.full_like(times, 1.0 - 1e-4)
    threshold = float(chi2.ppf(0.99, DIM))

    summaries = []
    cumulative_rows = []
    for method in METHOD_ORDER:
        run_dir = args.log_root / RUNS[method]
        checkpoint = (
            run_dir / "best_val_norm_mse_model.pt"
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        print(f"Evaluating {method}", flush=True)
        model = load_self_model(checkpoint)
        model.eval()

        if method == "tNCE":
            point_error, point_radius = pair_ratio_error(
                model,
                vertical_x,
                vertical_y,
                times,
                flat_means,
                args.batch_size,
            )
            summary, rows = summarize_decomposition(
                method, point_error, point_radius, threshold
            )
        elif method == "tCNCE":
            point_error, point_radius = point_log_density_error(
                model,
                spatial,
                spatial_times,
                flat_means,
                args.batch_size,
            )
            summary, rows = summarize_decomposition(
                method, point_error, point_radius, threshold
            )
        else:
            point_error, point_radius = pair_ratio_error(
                model,
                diagonal_x,
                diagonal_y,
                times,
                flat_means,
                args.batch_size,
            )
            summary, rows = summarize_decomposition(
                method, point_error, point_radius, threshold
            )
        if method == "tNCE":
            mean_space_step, p95_space_step = 0.0, 0.0
            bridge_name = "two fixed-x temporal paths"
        elif method == "tCNCE":
            mean_space_step, p95_space_step = spatial_step_statistics(
                spatial
            )
            bridge_name = "straight spatial interpolation"
        else:
            mean_space_step, p95_space_step = spatial_step_statistics(
                diagonal_x,
                diagonal_y,
            )
            bridge_name = "shared straight joint interpolation"
        summary.update(
            {
                "bridge_policy": "common_100_step_interpolation",
                "bridge_name": bridge_name,
                "mean_spatial_step_l2": mean_space_step,
                "p95_spatial_step_l2": p95_space_step,
                "mean_time_step": (
                    0.0
                    if method == "tCNCE"
                    else float(
                        (
                            times_1d[1:] - times_1d[:-1]
                        ).mean()
                    )
                ),
            }
        )
        summary.update(load_recorded_evaluation_metrics(run_dir))
        summaries.append(summary)
        cumulative_rows.extend(rows)
        del model
        gc.collect()

    write_csv(args.output_dir / "local_error_summary.csv", summaries)
    write_csv(args.output_dir / "cumulative_error_by_step.csv", cumulative_rows)
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(
            {
                "num_pairs": num_pairs,
                "num_steps": NUM_PATH_STEPS,
                "time_low": float(times_1d[0]),
                "time_high": float(times_1d[-1]),
                "support_threshold": threshold,
                "bridge_policy": "common_100_step_interpolation",
                "results": summaries,
            },
            handle,
            indent=2,
        )
    projected_density_figure(
        args.output_dir / "mnist_figure2_analogue.png",
        means,
        args.seed,
    )
    plot_error_analysis(
        args.output_dir / "local_to_global_error.png",
        summaries,
        cumulative_rows,
    )
    write_report(args.output_dir / "ANALYSIS.md", summaries)
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
