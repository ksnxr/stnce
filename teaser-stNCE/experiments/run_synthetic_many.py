"""Train the teaser experiment grid and save per-run metrics as pickle files."""

import argparse
import logging
import pathlib
import pickle
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

LOSS_NAME = "tnce"
SIGMA = 0.1
MU_1 = -5
MU_2 = 5
N_EPOCHS = 10000
EVAL_EVERY_EPOCHS = 200
EMA_DECAY = 0.9999

NITER = 5
BATCH_SIZE = 256
TRAIN_SIZE = 10_000
VAL_SIZE = 2_000
LOSSES = ["tnce", "tcnce", "stnce_white"]
SELECTED_MU_PAIRS = [
    (-0.899666, -0.899666),
    (0.785953, 0.852597),
    (0.010033, 0.210031),
    (0.010033, 0.610027),
    (-0.913043, 0.906064),
    (-0.953177, -0.953177),
    (-0.966555, -0.900225),
    (0.799331, 0.998054),
    (-1.100334, -0.496941),
    (-1.073579, 1.419615),
    (1.000000, 1.000000),
    (-1.033445, -0.965996),
    (-1.100334, -0.894632),
    (-1.301003, -0.696849),
    (-2.290970, 1.548965),
    (-1.046823, -1.046823),
    (-1.100334, -1.031767),
    (1.000000, 1.200669),
    (-1.501672, -0.899442),
    (-3.000000, -0.371238),
    (-3.000000, -3.000000),
    (-3.000000, -2.932612),
    (-3.000000, -2.798280),
    (-3.000000, -2.392284),
    (-3.000000, -1.073579),
]


ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ebmlearn.default import RESULTS_FOLDER
from ebmlearn.loss import stnce_white_loss, tcnce_loss, tnce_loss
from ebmlearn.network import EnergyNet

logger = logging.getLogger(__name__)


def data_multimodality(mu_1: float, mu_2: float, sigma: float) -> float:
    return (mu_2 - mu_1) / (mu_2 - mu_1 + 2.0 * sigma)


def data_support_mismatch(mu_1: float, mu_2: float, sigma: float) -> float:
    a = min(mu_1 - sigma, mu_2 - sigma)
    b = max(mu_1 + sigma, mu_2 + sigma)
    overlap = max(0.0, min(b, 1.0) - max(a, -1.0))
    return 1.0 - overlap / (b - a)


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow_params = [
            parameter.detach().clone()
            for parameter in model.parameters()
            if parameter.requires_grad
        ]

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        shadow_index = 0
        for parameter in model.parameters():
            if not parameter.requires_grad:
                continue
            self.shadow_params[shadow_index].mul_(self.decay).add_(
                parameter.detach(), alpha=1.0 - self.decay
            )
            shadow_index += 1

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        shadow_index = 0
        for parameter in model.parameters():
            if not parameter.requires_grad:
                continue
            parameter.copy_(self.shadow_params[shadow_index])
            shadow_index += 1


def normal_log_prob(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    variance = std**2
    normalizer = torch.log(
        torch.tensor(2.0 * torch.pi * variance, device=x.device, dtype=x.dtype)
    )
    return -0.5 * (((x - mean) ** 2) / variance + normalizer)


def make_data_log_density(mu_1: float, mu_2: float, sigma: float):
    def data_log_density(data_samples: torch.Tensor) -> torch.Tensor:
        component_log_probs = torch.stack(
            [
                normal_log_prob(data_samples, mean=mu_1, std=sigma),
                normal_log_prob(data_samples, mean=mu_2, std=sigma),
            ],
            dim=0,
        )
        mixture_log_probs = torch.logsumexp(
            component_log_probs
            - torch.log(
                torch.tensor(2.0, device=data_samples.device, dtype=data_samples.dtype)
            ),
            dim=0,
        )
        return mixture_log_probs.squeeze(-1)

    return data_log_density


def sample_gaussian_mixture(
    num_samples: int,
    mu_1: float,
    mu_2: float,
    sigma: float,
    device: torch.device,
) -> torch.Tensor:
    component_probs = torch.full((2,), 0.5, device=device)
    component_ids = torch.multinomial(component_probs, num_samples, replacement=True)
    means = torch.tensor([[mu_1], [mu_2]], dtype=torch.float32, device=device)
    return means[component_ids] + sigma * torch.randn(num_samples, 1, device=device)


def create_dataset(
    num_samples: int,
    mu_1: float,
    mu_2: float,
    sigma: float,
    device: torch.device,
) -> TensorDataset:
    clean = sample_gaussian_mixture(num_samples, mu_1, mu_2, sigma, device=device)
    return TensorDataset(clean.cpu())


def get_loss_fn(loss_name: str):


    loss_fns = {
        "tnce": tnce_loss,
        "tcnce": tcnce_loss,
        "stnce_white": stnce_white_loss,
    }
    if loss_name not in loss_fns:
        raise ValueError(
            f"Unknown LOSS_NAME '{loss_name}'. Expected one of {tuple(loss_fns)}."
        )
    return loss_fns[loss_name]


def run_epoch(
    dataloader: DataLoader,
    model: EnergyNet,
    loss_fn,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    ema: ExponentialMovingAverage | None = None,
) -> float:
    is_training = optimizer is not None
    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_examples = 0

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for batch in dataloader:
            clean_data = batch[0]
            clean_data = clean_data.to(device)

            loss = loss_fn(x1=clean_data, energy_fn=model)

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if ema is not None:
                    ema.update(model)

            batch_size = clean_data.shape[0]
            total_loss += loss.item() * batch_size
            total_examples += batch_size

    return total_loss / total_examples


@torch.no_grad()
def evaluate_density_correlation(
    data_log_density_fn,
    model_log_density_fn,
    dataloader: DataLoader,
    device: torch.device,
    data_log_densities: torch.Tensor | None = None,
) -> float:
    if data_log_densities is None:
        data_log_densities = compute_log_densities_from_loader(
            density_fn=data_log_density_fn,
            dataloader=dataloader,
            device=device,
        )
    else:
        data_log_densities = data_log_densities.reshape(-1)
    model_log_densities = compute_log_densities_from_loader(
        density_fn=model_log_density_fn,
        dataloader=dataloader,
        device=device,
    )
    data_log_densities = data_log_densities.to(model_log_densities.device)

    finite_mask = torch.isfinite(data_log_densities) & torch.isfinite(
        model_log_densities
    )
    if not torch.all(finite_mask):
        num_nonfinite = finite_mask.numel() - torch.count_nonzero(finite_mask).item()
        raise ValueError(f"Found {num_nonfinite} non-finite log density value(s).")
    data_log_densities = data_log_densities.to(dtype=torch.float64)
    model_log_densities = model_log_densities.to(dtype=torch.float64)

    # R2 is invariant to independently scaling the predictor and response.
    # Shift before exponentiating so large log-density differences cannot
    # overflow into inf and poison the least-squares sums.
    data_densities = torch.exp(data_log_densities - torch.max(data_log_densities))
    model_densities = torch.exp(model_log_densities - torch.max(model_log_densities))

    design = torch.stack([data_densities, torch.ones_like(data_densities)], dim=1)
    solution = torch.linalg.lstsq(design, model_densities.unsqueeze(1)).solution
    fitted = (design @ solution).squeeze(1)
    residual_sum_squares = torch.sum((model_densities - fitted) ** 2)
    total_sum_squares = torch.sum(
        (model_densities - model_densities.mean()) ** 2
    ).clamp_min(torch.finfo(model_densities.dtype).eps)
    r_squared = 1.0 - residual_sum_squares / total_sum_squares
    return r_squared.item()


@torch.no_grad()
def compute_log_densities_from_loader(
    density_fn,
    dataloader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    log_densities = []
    for batch in dataloader:
        data = batch[0] if isinstance(batch, (tuple, list)) else batch
        data = data.to(device)
        log_densities.append(density_fn(data).reshape(-1).cpu())
    return torch.cat(log_densities, dim=0)


def run_training(
    mu_1: float,
    mu_2: float,
    sigma: float,
    loss_name: str,
    device: torch.device,
) -> dict[str, float]:
    torch.manual_seed(0)

    loss_fn = get_loss_fn(loss_name)
    data_log_density = make_data_log_density(mu_1, mu_2, sigma)

    train_dataset = create_dataset(TRAIN_SIZE, mu_1, mu_2, sigma, device=device)
    val_dataset = create_dataset(VAL_SIZE, mu_1, mu_2, sigma, device=device)
    test_dataset = create_dataset(VAL_SIZE, mu_1, mu_2, sigma, device=device)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_data_log_densities = compute_log_densities_from_loader(
        density_fn=data_log_density,
        dataloader=val_loader,
        device=device,
    )
    test_data_log_densities = compute_log_densities_from_loader(
        density_fn=data_log_density,
        dataloader=test_loader,
        device=device,
    )

    model = EnergyNet(data_dim=1, hidden_dim=128, num_hidden_layers=3).to(device)
    use_ema = EMA_DECAY > 0.0
    ema_model = (
        EnergyNet(data_dim=1, hidden_dim=128, num_hidden_layers=3).to(device)
        if use_ema
        else None
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    ema = ExponentialMovingAverage(model, decay=EMA_DECAY) if use_ema else None

    logger.info("Using device: %s", device)
    logger.info("Using loss: %s", loss_name)
    logger.info("Data multimodality: %.6f", data_multimodality(mu_1, mu_2, sigma))
    logger.info("Data support mismatch: %.6f", data_support_mismatch(mu_1, mu_2, sigma))
    logger.info("Using EMA decay: %s", EMA_DECAY if use_ema else "off")

    eval_model = model

    def model_log_density_at(
        data_samples: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        return -eval_model(data_samples, t)

    def model_log_density(data_samples: torch.Tensor) -> torch.Tensor:
        eval_model.eval()
        data_t = torch.ones(
            data_samples.shape[0],
            device=data_samples.device,
            dtype=data_samples.dtype,
        )
        if loss_name == "tnce":
            noise_t = torch.zeros_like(data_t)
            base_log_density = normal_log_prob(data_samples, mean=0.0, std=1.0).squeeze(
                -1
            )
            return (
                base_log_density
                + model_log_density_at(data_samples, data_t)
                - model_log_density_at(data_samples, noise_t)
            )

        return model_log_density_at(data_samples, data_t)

    history = []
    final_metrics = {}
    best_val_corr = float("-inf")
    for epoch in range(1, N_EPOCHS + 1):
        train_loss = run_epoch(train_loader, model, loss_fn, optimizer, device, ema=ema)
        should_eval = epoch % EVAL_EVERY_EPOCHS == 0 or epoch == N_EPOCHS
        if not should_eval:
            continue

        if use_ema:
            assert ema is not None
            assert ema_model is not None
            ema_model.load_state_dict(model.state_dict())
            ema.copy_to(ema_model)
            eval_model = ema_model
        else:
            eval_model = model
        eval_model.eval()
        val_loss = run_epoch(
            val_loader, eval_model, loss_fn, optimizer=None, device=device
        )
        test_loss = run_epoch(
            test_loader, eval_model, loss_fn, optimizer=None, device=device
        )
        val_corr = evaluate_density_correlation(
            data_log_density_fn=data_log_density,
            model_log_density_fn=model_log_density,
            dataloader=val_loader,
            device=device,
            data_log_densities=val_data_log_densities,
        )
        test_corr = evaluate_density_correlation(
            data_log_density_fn=data_log_density,
            model_log_density_fn=model_log_density,
            dataloader=test_loader,
            device=device,
            data_log_densities=test_data_log_densities,
        )
        step_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "test_loss": test_loss,
            "val_density_r2": val_corr,
            "test_density_r2": test_corr,
            "evaluate_density_r2": test_corr,
        }
        history.append(step_metrics)
        if val_corr > best_val_corr:
            best_val_corr = val_corr
            final_metrics = {
                "data_multimodality": data_multimodality(mu_1, mu_2, sigma),
                "data_support_mismatch": data_support_mismatch(mu_1, mu_2, sigma),
                "evaluate_density_r2": test_corr,
                "best_epoch": epoch,
                "best_val_density_r2": val_corr,
                "test_density_r2": test_corr,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "test_loss": test_loss,
            }
        logger.info(
            "Epoch %03d/%d - train_loss: %.6f - val_loss: %.6f - "
            "test_loss: %.6f - val_density_r2: %.6f - test_density_r2: %.6f",
            epoch,
            N_EPOCHS,
            train_loss,
            val_loss,
            test_loss,
            val_corr,
            test_corr,
        )

    logger.info(
        "Selected epoch %d by highest val_density_r2 (%.6f); reporting "
        "test_density_r2=%.6f",
        final_metrics["best_epoch"],
        final_metrics["best_val_density_r2"],
        final_metrics["test_density_r2"],
    )
    return {
        **final_metrics,
        "last_epoch": history[-1]["epoch"],
        "last_val_density_r2": history[-1]["val_density_r2"],
        "last_test_density_r2": history[-1]["test_density_r2"],
        "history": history,
    }


def train_single_limitation_run(
    mu_1: float,
    mu_2: float,
    loss_name: str,
) -> dict[str, float]:
    return run_training(
        mu_1=mu_1,
        mu_2=mu_2,
        sigma=SIGMA,
        loss_name=loss_name,
        device=torch.device("cpu"),
    )


def loss_slug(loss_name: str) -> str:
    return loss_name.replace("+", "_plus_")


def save_results_pickle(
    result_payload: dict,
    loss_name: str,
    mu_pair_index: int,
) -> pathlib.Path:
    output_path = RESULTS_FOLDER / (
        f"run_synthetic_many_results_{loss_slug(loss_name)}"
        f"_pair_{mu_pair_index}.pkl"
    )

    with output_path.open("wb") as f:
        pickle.dump(result_payload, f)

    return output_path


def parse_mu_pair_indices(value: str) -> list[int]:
    if value == "all":
        return list(range(len(SELECTED_MU_PAIRS)))

    indices = []
    seen = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            raise argparse.ArgumentTypeError(
                "mu pair indices cannot contain empty entries."
            )

        if "-" in part:
            start_text, end_text = part.split("-", maxsplit=1)
            try:
                start = int(start_text)
                end = int(end_text)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"Invalid mu pair range {part!r}; expected start-end."
                ) from exc
            if start > end:
                raise argparse.ArgumentTypeError(
                    f"Invalid mu pair range {part!r}; start must be <= end."
                )
            part_indices = range(start, end + 1)
        else:
            try:
                part_indices = [int(part)]
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"Invalid mu pair index {part!r}; expected an integer."
                ) from exc

        for index in part_indices:
            if index < 0 or index >= len(SELECTED_MU_PAIRS):
                raise argparse.ArgumentTypeError(
                    f"Mu pair index {index} is out of range; "
                    f"expected 0-{len(SELECTED_MU_PAIRS) - 1}."
                )
            if index not in seen:
                seen.add(index)
                indices.append(index)

    return indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run synthetic experiments sequentially for one loss, or for all losses "
            "one loss at a time."
        )
    )
    parser.add_argument(
        "--loss",
        choices=[*LOSSES, "all"],
        default="all",
        help="Loss to run. Use 'all' to generate one pickle per run.",
    )
    parser.add_argument(
        "--mu-pair-indices",
        type=parse_mu_pair_indices,
        default=parse_mu_pair_indices("all"),
        help=(
            "Zero-based entries of SELECTED_MU_PAIRS to run. Accepts 'all', "
            "comma-separated indices, and inclusive ranges, e.g. '0,3,7-9'."
        ),
    )
    return parser.parse_args()


def run_loss_experiments(
    loss_name: str,
    mu_pair_indices: list[int] | None = None,
) -> list[pathlib.Path]:
    if mu_pair_indices is None:
        mu_pair_indices = list(range(len(SELECTED_MU_PAIRS)))

    output_paths = []

    logger.info(
        "Running %d sequential experiments for loss=%s",
        len(mu_pair_indices),
        loss_name,
    )
    for mu_pair_index in mu_pair_indices:
        mu_1, mu_2 = SELECTED_MU_PAIRS[mu_pair_index]
        run_results = train_single_limitation_run(mu_1, mu_2, loss_name)

        best_metrics = {
            "epoch": run_results["best_epoch"],
            "train_loss": run_results["train_loss"],
            "val_loss": run_results["val_loss"],
            "test_loss": run_results["test_loss"],
            "val_density_r2": run_results["best_val_density_r2"],
            "test_density_r2": run_results["test_density_r2"],
            "evaluate_density_r2": run_results["evaluate_density_r2"],
        }
        last_metrics = run_results["history"][-1]
        result_payload = {
            "loss_name": loss_name,
            "mu_pair_index": mu_pair_index,
            "mu_1": mu_1,
            "mu_2": mu_2,
            "sigma": SIGMA,
            "n_epochs": N_EPOCHS,
            "eval_every_epochs": EVAL_EVERY_EPOCHS,
            "ema_decay": EMA_DECAY,
            "data_multimodality": run_results["data_multimodality"],
            "data_support_mismatch": run_results["data_support_mismatch"],
            "best_metrics": best_metrics,
            "last_metrics": last_metrics,
            "history": run_results["history"],
        }
        output_path = save_results_pickle(result_payload, loss_name, mu_pair_index)
        output_paths.append(output_path)

        logger.info(
            "loss=%s mu_pair_index=%d mu_1=%.3f mu_2=%.3f "
            "multimodality=%.6f support_mismatch=%.6f best_epoch=%d "
            "best_val_density_r2=%.6f best_test_density_r2=%.6f "
            "last_epoch=%d last_val_density_r2=%.6f last_test_density_r2=%.6f",
            loss_name,
            mu_pair_index,
            mu_1,
            mu_2,
            run_results["data_multimodality"],
            run_results["data_support_mismatch"],
            run_results["best_epoch"],
            run_results["best_val_density_r2"],
            run_results["test_density_r2"],
            last_metrics["epoch"],
            last_metrics["val_density_r2"],
            last_metrics["test_density_r2"],
        )
        logger.info("Saved pickle to %s", output_path)

    return output_paths


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    args = parse_args()
    loss_names = LOSSES if args.loss == "all" else [args.loss]
    for loss_name in loss_names:
        run_loss_experiments(loss_name, args.mu_pair_indices)


if __name__ == "__main__":
    main()
