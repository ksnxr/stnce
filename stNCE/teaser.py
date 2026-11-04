"""Teaser experiment for CondOT parameter recovery in a 1D two-mode GMM.

The script compares the TNCE, TCNCE, and TSTNCE objectives implemented in
``losses_lib.py`` on a tiny parametric problem. The target distribution is a
two-component Gaussian mixture with fixed component standard deviation within
each grid cell; the learned parameters are the means, standard deviations, and
left-component mixture weight at every discrete time.

The learned energy is time-conditioned: for a discrete grid of ``num_times``
CondOT times in [0, 1], it stores a separate two-component GMM parameterization.
Training batches only use times from this grid. Evaluation reports the
parameters at the final grid point, t = 1, corresponding to the data
distribution.

The standard normal base distribution has full mathematical support, so this
teaser uses a configurable central interval as its "practical support". For
each requested retained-mass value and multimodality value, the component
standard deviation is chosen so that approximately that much probability mass
lies in the practical support. The component means are fixed symmetrically around
zero with distance = multimodality * std, which is Ashman's D for equal-variance
components. The TCNCE and TSTNCE white-noise kernels use that same std. All
target samples are used for estimation, including samples outside that practical
support interval.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/st_nce_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/st_nce_xdg_cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize
from torch.nn.utils import parameters_to_vector, vector_to_parameters

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import losses_lib
import prob_paths_lib


INIT_COMPONENT_STD = 0.1
MIN_COMPONENT_STD = 1e-4
DEFAULT_METHODS = ("tnce", "tcnce", "tstnce")


class TimeGridTwoModeGMMEnergy(nn.Module):
    """Time-grid energy for normalized 1D two-component GMMs."""

    def __init__(
        self,
        init_left: float,
        init_right: float,
        init_std: float,
        init_weight_left: float,
        num_times: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()
        self.num_times = num_times
        time_grid = torch.linspace(0.0, 1.0, num_times, dtype=dtype, device=device)
        self.register_buffer("time_grid", time_grid)

        init_weight_left = float(np.clip(init_weight_left, 1e-4, 1.0 - 1e-4))
        init_final_means = torch.tensor(
            [init_left, init_right], dtype=dtype, device=device
        )
        init_means = time_grid[:, None] * init_final_means[None, :]
        init_path_stds = torch.sqrt(
            (time_grid[:, None] * init_std).square()
            + (1.0 - time_grid[:, None]).square()
        ).expand(num_times, 2)
        init_path_stds = torch.clamp(init_path_stds, min=2.0 * MIN_COMPONENT_STD)

        self.means_table = nn.Parameter(init_means)
        self.raw_std = nn.Parameter(
            softplus_inverse(init_path_stds - MIN_COMPONENT_STD)
        )
        self.logit_left = nn.Parameter(
            torch.full(
                (num_times,),
                math.log(init_weight_left / (1.0 - init_weight_left)),
                dtype=dtype,
                device=device,
            )
        )

    def component_stds(self) -> torch.Tensor:
        return F.softplus(self.raw_std) + MIN_COMPONENT_STD

    def weight_lefts(self) -> torch.Tensor:
        return torch.sigmoid(self.logit_left)

    def time_indices(
        self, t: torch.Tensor | None, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        if t is None:
            return torch.full(
                (batch_size,), self.num_times - 1, dtype=torch.long, device=device
            )
        t = t.reshape(-1).to(device=device, dtype=self.time_grid.dtype)
        return (
            torch.round(t * (self.num_times - 1))
            .to(torch.long)
            .clamp(0, self.num_times - 1)
        )

    def log_prob(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        x = x.reshape(-1, 1)
        time_idx = self.time_indices(t, batch_size=x.shape[0], device=x.device)
        means = self.means_table[time_idx]
        stds = self.component_stds()[time_idx]
        weight_left = self.weight_lefts()[time_idx]
        min_weight = torch.finfo(x.dtype).eps
        weights = torch.stack([weight_left, 1.0 - weight_left], dim=1).clamp_min(
            min_weight
        )
        log_weights = torch.log(weights / weights.sum(dim=1, keepdim=True))
        z = (x - means) / stds
        component_log_probs = (
            -0.5 * z.square() - torch.log(stds) - 0.5 * math.log(2.0 * math.pi)
        )
        return torch.logsumexp(log_weights + component_log_probs, dim=1, keepdim=True)

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        return -self.log_prob(x, t)

    @torch.no_grad()
    def data_time_estimates(self) -> tuple[float, float, float, float, float]:
        means = self.means_table[-1].detach().cpu().numpy()
        stds = self.component_stds()[-1].detach().cpu().numpy()
        weights = np.array(
            [
                float(self.weight_lefts()[-1].detach().cpu()),
                float(1.0 - self.weight_lefts()[-1].detach().cpu()),
            ]
        )
        order = np.argsort(means)
        means = means[order]
        stds = stds[order]
        weights = weights[order]
        return (
            float(means[0]),
            float(means[1]),
            float(stds[0]),
            float(stds[1]),
            float(weights[0]),
        )


def softplus_inverse(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def parse_grid(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def white_noise_type(component_std: float) -> str:
    return f"white-{component_std:.17g}"


def component_support_mass(
    mean: float, component_std: float, support_radius: float
) -> float:
    upper = (support_radius - mean) / component_std
    lower = (-support_radius - mean) / component_std
    return normal_cdf(upper) - normal_cdf(lower)


def distance_from_multimodality(multimodality: float, component_std: float) -> float:
    return multimodality * component_std


def mixture_support_mass(
    multimodality: float,
    component_std: float,
    weight_left: float,
    support_radius: float,
) -> float:
    distance = distance_from_multimodality(multimodality, component_std)
    left = -0.5 * distance
    right = 0.5 * distance
    return weight_left * component_support_mass(left, component_std, support_radius) + (
        1.0 - weight_left
    ) * component_support_mass(right, component_std, support_radius)


def params_for_retained_mass_and_multimodality(
    target_mass: float,
    multimodality: float,
    weight_left: float,
    support_radius: float,
) -> tuple[float, float, float, float]:
    max_mass = mixture_support_mass(
        multimodality, MIN_COMPONENT_STD, weight_left, support_radius
    )
    if target_mass >= max_mass:
        component_std = MIN_COMPONENT_STD
        distance = distance_from_multimodality(multimodality, component_std)
        return -0.5 * distance, 0.5 * distance, component_std, max_mass

    lo = MIN_COMPONENT_STD
    hi = INIT_COMPONENT_STD
    while (
        mixture_support_mass(multimodality, hi, weight_left, support_radius)
        > target_mass
    ):
        hi *= 2.0

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if (
            mixture_support_mass(multimodality, mid, weight_left, support_radius)
            > target_mass
        ):
            lo = mid
        else:
            hi = mid

    component_std = 0.5 * (lo + hi)
    actual_mass = mixture_support_mass(
        multimodality, component_std, weight_left, support_radius
    )
    distance = distance_from_multimodality(multimodality, component_std)
    return -0.5 * distance, 0.5 * distance, component_std, actual_mass


def sample_two_mode_gmm(
    num_samples: int,
    left_mean: float,
    right_mean: float,
    component_std: float,
    weight_left: float,
    generator: torch.Generator,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    components = (
        torch.rand(num_samples, generator=generator, device=device) >= weight_left
    )
    means = torch.where(
        components,
        torch.tensor(right_mean, dtype=dtype, device=device),
        torch.tensor(left_mean, dtype=dtype, device=device),
    )
    samples = means + component_std * torch.randn(
        num_samples, generator=generator, dtype=dtype, device=device
    )
    return samples.reshape(-1, 1)


def make_config(
    device: torch.device,
    args: argparse.Namespace,
    component_std: float,
) -> SimpleNamespace:
    noise_type = white_noise_type(component_std)
    return SimpleNamespace(
        device=device,
        data=SimpleNamespace(dim=1, is_image=False),
        estim=SimpleNamespace(
            prob_path="ot",
            param="energy",
            num_noise_classes=float("inf"),
            num_cla_classes=2,
            normalize_time=False,
            cnce_noise_type=noise_type,
            stnce_noise_type=noise_type,
            score_type=args.score_type,
            time_score_type=args.time_score_type,
            time_std=args.time_std,
            rne=args.rne,
            rne_reference=args.rne_reference,
            resolve_collisions=args.resolve_collisions,
            eps1=args.eps1,
            eps2=args.eps2,
            prob_path_eps=args.prob_path_eps,
            t_dist=args.t_dist,
            ctsm_c=1.0,
        ),
    )


def build_condot_path(
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    true_left: float,
    true_right: float,
    true_std: float,
):
    data_statistics = {
        "means": torch.tensor([[true_left], [true_right]], dtype=dtype, device=device),
        "std": true_std,
    }
    return prob_paths_lib.get_prob_path(
        config=make_config(device=device, args=args, component_std=true_std),
        device=device,
        data_statistics=data_statistics,
    )


def discrete_time_grid(
    args: argparse.Namespace, device: torch.device, dtype: torch.dtype
):
    return torch.linspace(0.0, 1.0, args.num_times, dtype=dtype, device=device)


def sample_grid_times(
    batch_size: int,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    grid = discrete_time_grid(args, device=device, dtype=dtype)
    indices = torch.randint(0, args.num_times, (batch_size,), device=device)
    return grid[indices]


def sample_grid_time_pairs(
    batch_size: int,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    grid = discrete_time_grid(args, device=device, dtype=dtype)
    indices = torch.randint(0, args.num_times, (batch_size,), device=device)
    directions = torch.where(
        torch.rand(batch_size, device=device) < 0.5,
        torch.full((batch_size,), -1, dtype=torch.long, device=device),
        torch.ones(batch_size, dtype=torch.long, device=device),
    )
    other_indices = indices + directions
    other_indices = torch.where(other_indices < 0, indices + 1, other_indices)
    other_indices = torch.where(
        other_indices >= args.num_times, indices - 1, other_indices
    )
    return grid[indices], grid[other_indices]


def build_discrete_loss_batch(
    method: str,
    data: torch.Tensor,
    prob_path,
    args: argparse.Namespace,
    true_std: float,
):
    batch_size = data.shape[0]
    device = data.device
    dtype = data.dtype

    if method == "tnce":
        ts, taus = sample_grid_time_pairs(batch_size, args, device=device, dtype=dtype)
        xs = prob_path.sample(data, ts)
        return xs, ts, taus

    if method == "tcnce":
        ts = sample_grid_times(batch_size, args, device=device, dtype=dtype)
        xs = prob_path.sample(data, ts)
        ys = xs + true_std * torch.randn_like(xs)
        conds = torch.zeros((batch_size, 1), dtype=dtype, device=device)
        return xs, ts, ys, ts, conds

    if method == "tstnce":
        ts, taus = sample_grid_time_pairs(batch_size, args, device=device, dtype=dtype)
        xs = prob_path.sample(data, ts)
        ys = xs + true_std * torch.randn_like(xs)
        conds = torch.zeros((batch_size, 1), dtype=dtype, device=device)
        return xs, ts, ys, taus, conds

    raise ValueError(f"Unknown method {method!r}.")


def build_loss_and_batch(
    method: str,
    data: torch.Tensor,
    prob_path,
    args: argparse.Namespace,
    true_std: float,
):
    loss_fn = losses_lib.get_loss_fn(
        estimator=method,
        reg_estimator="none",
        num_noise_classes=float("inf"),
        num_cla_classes=2,
        param="energy",
        rne=args.rne,
    )
    return loss_fn, build_discrete_loss_batch(
        method=method,
        data=data,
        prob_path=prob_path,
        args=args,
        true_std=true_std,
    )


def init_candidates(
    data: torch.Tensor,
    rng: np.random.Generator,
    num_restarts: int,
) -> list[tuple[float, float, float, float]]:
    values = data.detach().cpu().numpy().reshape(-1)
    if values.size == 0:
        return [(-0.5, 0.5, INIT_COMPONENT_STD, 0.5)]

    q25, q50, q75 = np.quantile(values, [0.25, 0.5, 0.75])
    spread = max(float(np.std(values)), 0.25)
    init_std = max(0.05, min(spread, 2.0))
    candidates = [
        (float(q25), float(q75), init_std, 0.5),
        (float(q50 - spread), float(q50 + spread), init_std, 0.5),
        (float(values.min()), float(values.max()), init_std, 0.5),
    ]

    while len(candidates) < num_restarts:
        midpoint = float(q50 + rng.normal(scale=0.5 * spread))
        gap = float(max(0.2, rng.lognormal(mean=math.log(spread + 0.2), sigma=0.5)))
        std = float(max(0.02, rng.lognormal(mean=math.log(init_std), sigma=0.4)))
        weight = float(rng.uniform(0.2, 0.8))
        candidates.append((midpoint - 0.5 * gap, midpoint + 0.5 * gap, std, weight))

    return candidates[:num_restarts]


def optimize_once(
    method: str,
    data: torch.Tensor,
    prob_path,
    true_left: float,
    true_right: float,
    true_std: float,
    true_weight_left: float,
    args: argparse.Namespace,
    rng: np.random.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float | int | bool | str]:
    loss_fn, batch = build_loss_and_batch(
        method=method,
        data=data,
        prob_path=prob_path,
        args=args,
        true_std=true_std,
    )

    best_result = None
    best_model = None
    for init_left, init_right, init_std, init_weight in init_candidates(
        data, rng=rng, num_restarts=args.restarts
    ):
        if init_right <= init_left:
            init_left, init_right = min(init_left, init_right), max(
                init_left, init_right
            )
            init_right = max(init_right, init_left + 1e-3)
        model = TimeGridTwoModeGMMEnergy(
            init_left=init_left,
            init_right=init_right,
            init_std=init_std,
            init_weight_left=init_weight,
            num_times=args.num_times,
            dtype=dtype,
            device=device,
        )
        params = list(model.parameters())

        def objective(x_np):
            x_tensor = torch.from_numpy(x_np).to(device=device, dtype=dtype)
            vector_to_parameters(x_tensor, params)
            model.zero_grad(set_to_none=True)
            loss = loss_fn(model, batch)
            if not torch.isfinite(loss):
                bad_grad = np.zeros_like(x_np, dtype=np.float64)
                return float("inf"), bad_grad
            loss.backward()
            grad = parameters_to_vector([p.grad for p in params])
            return (
                float(loss.detach().cpu()),
                grad.detach().cpu().numpy().astype(np.float64),
            )

        x0 = parameters_to_vector(params).detach().cpu().numpy().astype(np.float64)
        result = minimize(
            objective,
            x0=x0,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": args.maxiter, "ftol": args.ftol, "gtol": args.gtol},
        )
        if best_result is None or result.fun < best_result.fun:
            best_result = result
            best_model = model
            vector_to_parameters(
                torch.from_numpy(result.x).to(device=device, dtype=dtype),
                best_model.parameters(),
            )

    assert best_result is not None and best_model is not None

    (
        est_left,
        est_right,
        est_left_std,
        est_right_std,
        est_weight_left,
    ) = best_model.data_time_estimates()
    mean_error = math.sqrt((est_left - true_left) ** 2 + (est_right - true_right) ** 2)
    std_error = math.sqrt(
        (est_left_std - true_std) ** 2 + (est_right_std - true_std) ** 2
    )
    weight_error = abs(est_weight_left - true_weight_left)
    total_error = math.sqrt(mean_error**2 + std_error**2 + weight_error**2)

    return {
        "success": bool(best_result.success),
        "message": str(best_result.message),
        "loss": float(best_result.fun),
        "nfev": int(best_result.nfev),
        "est_left": est_left,
        "est_right": est_right,
        "est_left_std": est_left_std,
        "est_right_std": est_right_std,
        "est_weight_left": est_weight_left,
        "mean_error": mean_error,
        "std_error": std_error,
        "weight_error": weight_error,
        "total_error": total_error,
    }


def run_grid(args: argparse.Namespace) -> list[dict[str, float | int | bool | str]]:
    device = torch.device(args.device)
    dtype = torch.float64 if args.float64 else torch.float32
    torch.manual_seed(args.seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    rows = []
    for multimodality in args.multimodality:
        for target_retained in args.retained:
            (
                left_mean,
                right_mean,
                component_std,
                actual_retained,
            ) = params_for_retained_mass_and_multimodality(
                target_retained,
                multimodality=multimodality,
                weight_left=args.weight_left,
                support_radius=args.support_radius,
            )
            distance = right_mean - left_mean
            prob_path = build_condot_path(
                args=args,
                device=device,
                dtype=dtype,
                true_left=left_mean,
                true_right=right_mean,
                true_std=component_std,
            )

            for repeat in range(args.repeats):
                data = sample_two_mode_gmm(
                    args.num_samples,
                    left_mean=left_mean,
                    right_mean=right_mean,
                    component_std=component_std,
                    weight_left=args.weight_left,
                    generator=generator,
                    dtype=dtype,
                    device=device,
                )
                for method in args.methods:
                    result = optimize_once(
                        method=method,
                        data=data,
                        prob_path=prob_path,
                        true_left=left_mean,
                        true_right=right_mean,
                        true_std=component_std,
                        true_weight_left=args.weight_left,
                        args=args,
                        rng=rng,
                        device=device,
                        dtype=dtype,
                    )
                    row = {
                        "method": method,
                        "multimodality": multimodality,
                        "true_distance": distance,
                        "target_retained": target_retained,
                        "actual_retained": actual_retained,
                        "num_samples": args.num_samples,
                        "num_times": args.num_times,
                        "repeat": repeat,
                        "true_left": left_mean,
                        "true_right": right_mean,
                        "true_std": component_std,
                        "white_noise_type": white_noise_type(component_std),
                        "true_weight_left": args.weight_left,
                    }
                    row.update(result)
                    rows.append(row)
                    print(
                        f"{method:5s} retained={actual_retained:.2f} "
                        f"multimodality={multimodality:.2f} "
                        f"distance={distance:.2f} std={component_std:.3f} repeat={repeat} "
                        f"error={result['total_error']:.4f} "
                        f"mixture_error={result['weight_error']:.4f} "
                        f"loss={result['loss']:.4f}",
                        flush=True,
                    )

    return rows


def write_csv(rows: list[dict[str, float | int | bool | str]], path: Path) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_errors(
    rows: list[dict[str, float | int | bool | str]],
    methods: list[str],
    retained: list[float],
    multimodalities: list[float],
) -> dict[str, np.ndarray]:
    errors = {
        method: np.full((len(multimodalities), len(retained)), np.nan, dtype=np.float64)
        for method in methods
    }
    for i, multimodality in enumerate(multimodalities):
        for j, target_retained in enumerate(retained):
            for method in methods:
                selected = [
                    float(row["total_error"])
                    for row in rows
                    if row["method"] == method
                    and math.isclose(float(row["multimodality"]), multimodality)
                    and math.isclose(float(row["target_retained"]), target_retained)
                ]
                if selected:
                    errors[method][i, j] = float(np.mean(selected))
    return errors


def plot_errors(
    rows: list[dict[str, float | int | bool | str]],
    args: argparse.Namespace,
    output_path: Path,
) -> None:
    errors = aggregate_errors(
        rows,
        methods=args.methods,
        retained=args.retained,
        multimodalities=args.multimodality,
    )
    finite_values = np.concatenate(
        [
            values[np.isfinite(values)]
            for values in errors.values()
            if np.isfinite(values).any()
        ]
    )
    vmax = float(np.percentile(finite_values, 95)) if finite_values.size else 1.0
    vmax = max(vmax, 1e-6)

    fig, axes = plt.subplots(
        1,
        len(args.methods),
        figsize=(4.6 * len(args.methods), 3.8),
        constrained_layout=True,
        squeeze=False,
    )

    def axis_extent(values: list[float]) -> tuple[float, float]:
        if len(values) == 1 or math.isclose(min(values), max(values)):
            value = values[0]
            return value - 0.5, value + 0.5
        return min(values), max(values)

    x_min, x_max = axis_extent(args.retained)
    y_min, y_max = axis_extent(args.multimodality)

    for ax, method in zip(axes[0], args.methods):
        image = ax.imshow(
            errors[method],
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            vmin=0.0,
            vmax=vmax,
            extent=[x_min, x_max, y_min, y_max],
        )
        ax.set_title(method.upper())
        ax.set_xlabel("Mass retained in base support")
        ax.set_ylabel("Multimodality (Ashman's D)")
        ax.set_xticks(args.retained)
        ax.set_yticks(args.multimodality)
        ax.tick_params(axis="x", labelrotation=35)
        fig.colorbar(image, ax=ax, label="parameter error")

    fig.suptitle(
        "CondOT 1D two-mode GMM parameter recovery "
        f"(std solved per cell, support=[-{args.support_radius}, {args.support_radius}])"
    )
    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=THIS_DIR / "teaser_outputs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--float64", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--num-times", type=int, default=11)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--maxiter", type=int, default=120)
    parser.add_argument("--ftol", type=float, default=1e-9)
    parser.add_argument("--gtol", type=float, default=1e-6)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--support-radius", type=float, default=2.0)
    parser.add_argument("--weight-left", type=float, default=0.35)
    parser.add_argument(
        "--retained",
        type=parse_grid,
        default=parse_grid("0.25,0.50,0.75,0.95"),
        help="Comma-separated target masses retained in the practical base support.",
    )
    parser.add_argument(
        "--multimodality",
        type=parse_grid,
        default=parse_grid("0.5,1.0,3.0,10.0"),
        help="Comma-separated standardized mode separations, |mu2-mu1| / std.",
    )
    parser.add_argument("--time-std", type=float, default=0.1)
    parser.add_argument("--eps1", type=float, default=1e-4)
    parser.add_argument("--eps2", type=float, default=1e-4)
    parser.add_argument("--prob-path-eps", type=float, default=1e-4)
    parser.add_argument(
        "--t-dist",
        type=str,
        choices=["uniform", "log-snr", "ctsm-is", "mle"],
        default="uniform",
    )
    parser.add_argument("--resolve-collisions", action="store_true")
    parser.add_argument("--rne", action="store_true")
    parser.add_argument("--rne-reference", action="store_true")
    parser.add_argument(
        "--score-type",
        type=str,
        choices=["none", "oracle", "flat", "gaussian"],
        default="none",
    )
    parser.add_argument(
        "--time-score-type",
        type=str,
        choices=["none", "oracle", "fp"],
        default="none",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=DEFAULT_METHODS,
        default=list(DEFAULT_METHODS),
    )
    args = parser.parse_args()

    args.retained = sorted(args.retained)
    args.multimodality = sorted(args.multimodality)
    if not 0.0 < args.weight_left < 1.0:
        raise ValueError("--weight-left must be between 0 and 1.")
    if args.support_radius <= 0.0:
        raise ValueError("--support-radius must be positive.")
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")
    if args.num_times < 2:
        raise ValueError("--num-times must be at least 2.")
    if any(value < 0.0 for value in args.multimodality):
        raise ValueError("--multimodality values must be nonnegative.")

    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = run_grid(args)
    csv_path = args.output_dir / "teaser_results.csv"
    figure_path = args.output_dir / "teaser_errors.png"
    write_csv(rows, csv_path)
    plot_errors(rows, args=args, output_path=figure_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
