"""16-mode CondOT teaser for TNCE, TCNCE, and TSTNCE.

This script compares the time-indexed contrastive objectives implemented in
``losses_lib.py`` on a 2D, 16-component Gaussian mixture. The target mixture
means live on a 4x4 grid. The model fixes the CondOT-path means and component
standard deviations to their known values and learns mixture weights across the
time grid. Evaluation reports the mixture-weight MSE at the clean-data time.

When ``num_times == 1``, the single time-grid point is the clean data time
``t = 1``. In that setting the TNCE, TCNCE, and TSTNCE choices are routed to the
ordinary NCE, CNCE, and STNCE losses, respectively.

Two target regimes are included:

* ``inside_separated``: component std is small relative to grid spacing, and the
  mode centers lie inside the practical support of the standard normal base.
* ``outside_broad``: component std is large relative to grid spacing, and the
  mode centers lie outside the practical support of the standard normal base.

All target samples are used for training. The practical base support is only
reported as a diagnostic; it does not filter samples.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
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
from scipy.optimize import minimize
from torch.nn.utils import parameters_to_vector, vector_to_parameters

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import losses_lib
import prob_paths_lib


GRID_SIDE = 4
NUM_COMPONENTS = GRID_SIDE * GRID_SIDE
DATA_DIM = 2
MIN_COMPONENT_STD = 1e-4
DEFAULT_METHODS = ("tnce", "tcnce", "tstnce")
DEFAULT_SCENARIOS = ("inside_separated", "outside_broad")


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    means: np.ndarray
    component_std: float
    weights: np.ndarray
    spacing: float
    mass_in_base_box: float

    @property
    def std_to_spacing(self) -> float:
        return self.component_std / self.spacing


class TimeGridGMMEnergy(nn.Module):
    """Time-grid GMM energy with fixed CondOT geometry and learned weights."""

    def __init__(
        self,
        true_means: np.ndarray,
        component_std: float,
        init_weights: np.ndarray,
        num_times: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()
        self.num_times = num_times
        self.num_components = true_means.shape[0]
        self.dim = true_means.shape[1]

        if num_times == 1:
            time_grid = torch.ones(1, dtype=dtype, device=device)
        else:
            time_grid = torch.linspace(0.0, 1.0, num_times, dtype=dtype, device=device)
        self.register_buffer("time_grid", time_grid)

        final_means = torch.as_tensor(true_means, dtype=dtype, device=device)
        final_stds = torch.full(
            (self.num_components,),
            float(component_std),
            dtype=dtype,
            device=device,
        ).clamp_min(2.0 * MIN_COMPONENT_STD)
        final_weights = torch.as_tensor(
            normalize_weights(init_weights), dtype=dtype, device=device
        )

        path_means = time_grid[:, None, None] * final_means[None, :, :]
        path_stds = torch.sqrt(
            torch.square(time_grid[:, None] * final_stds[None, :])
            + torch.square(1.0 - time_grid[:, None])
        ).clamp_min(2.0 * MIN_COMPONENT_STD)

        self.register_buffer("means_table", path_means)
        self.register_buffer("stds_table", path_stds)
        init_logits = torch.log(final_weights.clamp_min(torch.finfo(dtype).eps))
        self.logits_table = nn.Parameter(
            init_logits.reshape(1, self.num_components)
            .expand(num_times, self.num_components)
            .clone()
        )

    def component_stds(self) -> torch.Tensor:
        return self.stds_table

    def component_weights(self) -> torch.Tensor:
        return torch.softmax(self.logits_table, dim=-1)

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
        x = x.reshape(x.shape[0], -1)
        time_idx = self.time_indices(t, batch_size=x.shape[0], device=x.device)
        means = self.means_table[time_idx]
        stds = self.component_stds()[time_idx]
        log_weights = torch.log_softmax(self.logits_table[time_idx], dim=-1)

        z = (x[:, None, :] - means) / stds[:, :, None]
        component_log_probs = (
            -0.5 * torch.sum(torch.square(z), dim=-1)
            - self.dim * torch.log(stds)
            - 0.5 * self.dim * math.log(2.0 * math.pi)
        )
        return torch.logsumexp(log_weights + component_log_probs, dim=1, keepdim=True)

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        return -self.log_prob(x, t)

    @torch.no_grad()
    def data_time_params(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        means = self.means_table[-1].detach().cpu().numpy()
        stds = self.component_stds()[-1].detach().cpu().numpy()
        weights = self.component_weights()[-1].detach().cpu().numpy()
        return means, stds, weights


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.clip(weights, 1e-8, None)
    return weights / weights.sum()


def white_noise_type(component_std: float) -> str:
    return f"white-{component_std:.17g}"


def component_box_mass(
    mean: np.ndarray, component_std: float, support_radius: float
) -> float:
    mass = 1.0
    for value in mean:
        upper = (support_radius - float(value)) / component_std
        lower = (-support_radius - float(value)) / component_std
        mass *= normal_cdf(upper) - normal_cdf(lower)
    return mass


def mixture_box_mass(
    means: np.ndarray,
    component_std: float,
    weights: np.ndarray,
    support_radius: float,
) -> float:
    masses = [component_box_mass(mean, component_std, support_radius) for mean in means]
    return float(np.dot(weights, np.asarray(masses)))


def make_grid_means(axis_values: np.ndarray) -> np.ndarray:
    xs, ys = np.meshgrid(axis_values, axis_values, indexing="xy")
    return np.stack([xs.reshape(-1), ys.reshape(-1)], axis=1).astype(np.float64)


def reference_weights() -> np.ndarray:
    grid_i, grid_j = np.meshgrid(np.arange(GRID_SIDE), np.arange(GRID_SIDE))
    scores = (
        0.45 * np.sin(0.9 * grid_i.reshape(-1))
        + 0.35 * np.cos(1.2 * grid_j.reshape(-1))
        + 0.08 * np.arange(NUM_COMPONENTS)
    )
    return normalize_weights(np.exp(scores))


def build_scenarios(args: argparse.Namespace) -> dict[str, Scenario]:
    weights = reference_weights()

    inside_axis = np.linspace(-args.inside_extent, args.inside_extent, GRID_SIDE)
    inside_spacing = float(inside_axis[1] - inside_axis[0])
    inside_means = make_grid_means(inside_axis)

    outside_axis = args.outside_start + args.outside_spacing * np.arange(GRID_SIDE)
    outside_means = make_grid_means(outside_axis)

    scenarios = {
        "inside_separated": Scenario(
            name="inside_separated",
            description="small std, separated modes, centers inside base support",
            means=inside_means,
            component_std=args.inside_std,
            weights=weights,
            spacing=inside_spacing,
            mass_in_base_box=mixture_box_mass(
                inside_means, args.inside_std, weights, args.support_radius
            ),
        ),
        "outside_broad": Scenario(
            name="outside_broad",
            description="large std, overlapping modes, centers outside base support",
            means=outside_means,
            component_std=args.outside_std,
            weights=weights,
            spacing=args.outside_spacing,
            mass_in_base_box=mixture_box_mass(
                outside_means, args.outside_std, weights, args.support_radius
            ),
        ),
    }
    return scenarios


def make_config(
    device: torch.device,
    args: argparse.Namespace,
    component_std: float,
) -> SimpleNamespace:
    noise_type = white_noise_type(component_std)
    return SimpleNamespace(
        device=device,
        data=SimpleNamespace(dim=DATA_DIM, is_image=False),
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
    scenario: Scenario,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
):
    data_statistics = {
        "means": torch.tensor(scenario.means, dtype=dtype, device=device),
        "std": scenario.component_std,
    }
    return prob_paths_lib.get_prob_path(
        config=make_config(
            device=device, args=args, component_std=scenario.component_std
        ),
        device=device,
        data_statistics=data_statistics,
    )


def discrete_time_grid(
    args: argparse.Namespace, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
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


def sample_gmm(
    num_samples: int,
    means: np.ndarray,
    component_std: float,
    weights: np.ndarray,
    generator: torch.Generator,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    means_t = torch.tensor(means, dtype=dtype, device=device)
    weights_t = torch.tensor(normalize_weights(weights), dtype=dtype, device=device)
    components = torch.multinomial(
        weights_t, num_samples=num_samples, replacement=True, generator=generator
    )
    noise = torch.randn(
        (num_samples, means.shape[1]), generator=generator, dtype=dtype, device=device
    )
    return means_t[components] + component_std * noise


def standard_normal_log_prob(x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(x.shape[0], -1)
    log_norm = -0.5 * x.shape[1] * math.log(2.0 * math.pi)
    return log_norm - 0.5 * torch.sum(torch.square(x), dim=1, keepdim=True)


def effective_estimator(method: str, num_times: int) -> str:
    if num_times != 1:
        return method
    return {
        "tnce": "nce",
        "tcnce": "cnce",
        "tstnce": "stnce",
    }[method]


def build_clean_loss_batch(method: str, data: torch.Tensor, noise_std: float):
    batch_size = data.shape[0]
    device = data.device
    dtype = data.dtype
    dim = data.reshape(batch_size, -1).shape[1]

    if method == "tnce":
        noise = torch.randn((batch_size, dim), dtype=dtype, device=device)
        return data, noise, standard_normal_log_prob

    if method == "tcnce":
        conditional_data = data + noise_std * torch.randn_like(data)
        conds = torch.zeros((batch_size, 1), dtype=dtype, device=device)
        return data, conditional_data, conds

    if method == "tstnce":
        conditional_noise = data + noise_std * torch.randn_like(data)
        data_cond = torch.zeros((batch_size, 1), dtype=dtype, device=device)
        noise = torch.randn((batch_size, dim), dtype=dtype, device=device)
        conditional_data = noise + noise_std * torch.randn_like(noise)
        noise_cond = torch.zeros((batch_size, 1), dtype=dtype, device=device)
        return (
            data,
            conditional_noise,
            data_cond,
            noise,
            conditional_data,
            noise_cond,
            standard_normal_log_prob,
        )

    raise ValueError(f"Unknown method {method!r}.")


def build_discrete_loss_batch(
    method: str,
    data: torch.Tensor,
    prob_path,
    args: argparse.Namespace,
    noise_std: float,
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
        ys = xs + noise_std * torch.randn_like(xs)
        conds = torch.zeros((batch_size, 1), dtype=dtype, device=device)
        return xs, ts, ys, ts, conds

    if method == "tstnce":
        ts, taus = sample_grid_time_pairs(batch_size, args, device=device, dtype=dtype)
        xs = prob_path.sample(data, ts)
        ys = prob_path.conditional_sample(xs, ts, taus)
        conds = prob_path.conditional_log_prob(xs, ts, ys, taus).reshape(-1, 1)
        return xs, ts, ys, taus, conds

    raise ValueError(f"Unknown method {method!r}.")


def build_loss_and_batch(
    method: str,
    data: torch.Tensor,
    prob_path,
    args: argparse.Namespace,
    noise_std: float,
):
    estimator = effective_estimator(method, args.num_times)
    loss_fn = losses_lib.get_loss_fn(
        estimator=estimator,
        reg_estimator="none",
        num_noise_classes=float("inf"),
        num_cla_classes=2,
        param="energy",
        rne=args.rne,
    )
    if args.num_times == 1:
        return loss_fn, build_clean_loss_batch(
            method=method,
            data=data,
            noise_std=noise_std,
        )
    return loss_fn, build_discrete_loss_batch(
        method=method,
        data=data,
        prob_path=prob_path,
        args=args,
        noise_std=noise_std,
    )


def initial_weights(
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> np.ndarray:
    logits = rng.normal(scale=args.init_logit_noise, size=NUM_COMPONENTS)
    return normalize_weights(np.exp(logits))


def evaluate_model(
    model: TimeGridGMMEnergy, scenario: Scenario
) -> dict[str, float | str]:
    _, _, est_weights = model.data_time_params()
    weight_deltas = est_weights - scenario.weights
    weight_mse = float(np.mean(np.square(weight_deltas)))
    return {
        "weight_mse": weight_mse,
    }


def optimize_once(
    method: str,
    data: torch.Tensor,
    prob_path,
    scenario: Scenario,
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
        noise_std=scenario.component_std,
    )

    best_result = None
    best_model = None
    for _ in range(args.restarts):
        init_weights = initial_weights(rng=rng, args=args)
        model = TimeGridGMMEnergy(
            true_means=scenario.means,
            component_std=scenario.component_std,
            init_weights=init_weights,
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
                return float("inf"), np.zeros_like(x_np, dtype=np.float64)
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
    metrics = evaluate_model(best_model, scenario=scenario)
    metrics.update(
        {
            "success": bool(best_result.success),
            "message": str(best_result.message),
            "loss": float(best_result.fun),
            "nfev": int(best_result.nfev),
        }
    )
    return metrics


def run_experiments(
    args: argparse.Namespace,
) -> list[dict[str, float | int | bool | str]]:
    device = torch.device(args.device)
    dtype = torch.float64 if args.float64 else torch.float32
    torch.manual_seed(args.seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    scenarios = build_scenarios(args)
    rows = []
    for scenario_name in args.scenarios:
        scenario = scenarios[scenario_name]
        prob_path = build_condot_path(
            scenario=scenario, args=args, device=device, dtype=dtype
        )
        for repeat in range(args.repeats):
            data = sample_gmm(
                num_samples=args.num_samples,
                means=scenario.means,
                component_std=scenario.component_std,
                weights=scenario.weights,
                generator=generator,
                dtype=dtype,
                device=device,
            )
            for method in args.methods:
                result = optimize_once(
                    method=method,
                    data=data,
                    prob_path=prob_path,
                    scenario=scenario,
                    args=args,
                    rng=rng,
                    device=device,
                    dtype=dtype,
                )
                row = {
                    "scenario": scenario.name,
                    "description": scenario.description,
                    "method": method,
                    "effective_estimator": effective_estimator(
                        method, args.num_times
                    ),
                    "repeat": repeat,
                    "num_samples": args.num_samples,
                    "num_times": args.num_times,
                    "component_std": scenario.component_std,
                    "spacing": scenario.spacing,
                    "std_to_spacing": scenario.std_to_spacing,
                    "mass_in_base_box": scenario.mass_in_base_box,
                    "support_radius": args.support_radius,
                    "white_noise_type": white_noise_type(scenario.component_std),
                    "true_min_weight": float(np.min(scenario.weights)),
                    "true_max_weight": float(np.max(scenario.weights)),
                }
                row.update(result)
                rows.append(row)
                print(
                    f"{method:6s} scenario={scenario.name:16s} repeat={repeat} "
                    f"estimator={row['effective_estimator']:5s} "
                    f"mass_in_base_box={scenario.mass_in_base_box:.3f} "
                    f"std/spacing={scenario.std_to_spacing:.2f} "
                    f"weight_mse={result['weight_mse']:.6f} "
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


def aggregate_metric(
    rows: list[dict[str, float | int | bool | str]],
    scenarios: list[str],
    methods: list[str],
    metric: str,
) -> np.ndarray:
    values = np.full((len(scenarios), len(methods)), np.nan, dtype=np.float64)
    for i, scenario in enumerate(scenarios):
        for j, method in enumerate(methods):
            selected = [
                float(row[metric])
                for row in rows
                if row["scenario"] == scenario and row["method"] == method
            ]
            if selected:
                values[i, j] = float(np.mean(selected))
    return values


def plot_summary(
    rows: list[dict[str, float | int | bool | str]],
    args: argparse.Namespace,
    output_path: Path,
) -> None:
    scenario_labels = {
        "inside_separated": "inside\nseparated",
        "outside_broad": "outside\nbroad",
    }
    colors = {
        "tnce": "#4C78A8",
        "tcnce": "#F58518",
        "tstnce": "#54A24B",
    }

    fig, ax = plt.subplots(figsize=(7.5, 4.4), constrained_layout=True)
    values = aggregate_metric(
        rows, scenarios=args.scenarios, methods=args.methods, metric="weight_mse"
    )
    x = np.arange(len(args.scenarios))
    width = 0.76 / max(1, len(args.methods))

    for j, method in enumerate(args.methods):
        offset = (j - 0.5 * (len(args.methods) - 1)) * width
        ax.bar(
            x + offset,
            values[:, j],
            width=width,
            label=effective_estimator(method, args.num_times).upper(),
            color=colors.get(method),
        )
    ax.set_title("mixture-weight MSE")
    ax.set_xticks(x)
    ax.set_xticklabels([scenario_labels.get(name, name) for name in args.scenarios])
    ax.set_ylabel("MSE")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=len(args.methods))
    title = "16-mode GMM mixture-weight recovery"
    if args.num_times > 1:
        title = "16-mode CondOT GMM time-grid mixture-weight recovery"
    fig.suptitle(title)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=THIS_DIR / "new_teaser_outputs"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--float64", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--num-times", type=int, default=11)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--ftol", type=float, default=1e-9)
    parser.add_argument("--gtol", type=float, default=1e-6)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--support-radius", type=float, default=2.0)
    parser.add_argument("--inside-extent", type=float, default=1.2)
    parser.add_argument("--inside-std", type=float, default=0.08)
    parser.add_argument("--outside-start", type=float, default=3.0)
    parser.add_argument("--outside-spacing", type=float, default=0.4)
    parser.add_argument("--outside-std", type=float, default=1.0)
    parser.add_argument("--init-logit-noise", type=float, default=0.35)
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
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=DEFAULT_SCENARIOS,
        default=list(DEFAULT_SCENARIOS),
    )
    args = parser.parse_args()

    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")
    if args.num_times < 1:
        raise ValueError("--num-times must be at least 1.")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive.")
    if args.restarts <= 0:
        raise ValueError("--restarts must be positive.")
    if args.support_radius <= 0.0:
        raise ValueError("--support-radius must be positive.")
    if args.inside_extent <= 0.0:
        raise ValueError("--inside-extent must be positive.")
    if args.inside_std <= 0.0 or args.outside_std <= 0.0:
        raise ValueError("Component stds must be positive.")
    if args.outside_spacing <= 0.0:
        raise ValueError("--outside-spacing must be positive.")
    if args.init_logit_noise < 0.0:
        raise ValueError("--init-logit-noise must be nonnegative.")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = run_experiments(args)
    csv_path = args.output_dir / "new_teaser_results.csv"
    figure_path = args.output_dir / "new_teaser_errors.png"
    write_csv(rows, csv_path)
    plot_summary(rows, args=args, output_path=figure_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
