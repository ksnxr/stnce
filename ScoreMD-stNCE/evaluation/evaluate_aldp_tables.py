import argparse
import os
from collections.abc import Sequence
from dataclasses import dataclass

from scoremd.data.dataset.aldp import ALDPDataset, CoarseGrainingLevel
from tqdm import tqdm
import jax
import numpy as onp
from scoremd.utils.evaluation import pairwise_distances, phi_psi_metrics, js_divergence
from scipy.stats import wasserstein_distance
import numpy as np
import jax.numpy as jnp


@dataclass
class DataSource:
    path: str

    def __post_init__(self):
        self.path = os.path.expanduser(self.path)


# Parent containing <dataset>_<method>_<weight>_<width>/<seed>/out/.
SAVED_OUTPUT_ROOT = "TODO-0"

i_values = ["0.01", "0.1", "1.0"]  # estimator_weights
j_values = ["0.001", "0.01", "0.1", "0.0"]  # time_stds
tcnce_alone_j_values = ["1.0", "0.1", "0.01"]  # time_stds

SOURCE_CONFIGS = {
    "stnce": ("STNCE", "stnce"),
    "tnce": ("TNCE", "tnce"),
    "stnce_alone": ("STNCE_ALONE", "stnce_alone"),
    "tnce_alone": ("TNCE_ALONE", "tnce_alone"),
    "tcnce_alone": ("TCNCE_ALONE", "tcnce_alone"),
}

ALONE_SOURCES = ["tnce_alone", "tcnce_alone", "stnce_alone"]

DATASET_CONFIGS = {
    "aldp": CoarseGrainingLevel.FULL,
    "aldp_full": CoarseGrainingLevel.NONE,
}


def get_i_values(source: str) -> list[str]:
    if source in {"stnce_alone", "tnce_alone", "tcnce_alone"}:
        return ["1.0"]
    return i_values


def get_j_values(source: str) -> list[str]:
    if source == "tcnce_alone":
        return tcnce_alone_j_values
    return j_values


def build_data_sources(
    sources: Sequence[str],
    dataset_name: str,
    estimator_weights: Sequence[str] | None = None,
    time_stds: Sequence[str] | None = None,
) -> dict[str, DataSource]:
    data_sources = {}
    for source in sources:
        label_prefix, directory_suffix = SOURCE_CONFIGS[source]
        for i in estimator_weights or get_i_values(source):
            for j in time_stds or get_j_values(source):
                data_sources[f"{label_prefix}_{i}_{j}"] = DataSource(
                    path=f"{SAVED_OUTPUT_ROOT}/{dataset_name}_{directory_suffix}_{i}_{j}",
                )
    return data_sources


def resolve_sources(args: argparse.Namespace) -> list[str]:
    if args.compare_alone:
        return ALONE_SOURCES

    sources = []
    if args.sources:
        sources.extend(args.sources)
    if args.source_flags:
        sources.extend(args.source_flags)

    return sources or ["stnce"]


def resolve_datasets(args: argparse.Namespace) -> list[str]:
    if args.dataset == "both":
        return list(DATASET_CONFIGS)
    return [args.dataset]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print ALDP evaluation tables for one output directory family."
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=sorted(SOURCE_CONFIGS),
        help="Source families to compare in the same table.",
    )
    parser.add_argument(
        "--compare-alone",
        action="store_true",
        help="Compare tnce_alone, tcnce_alone, and stnce_alone in the same table.",
    )
    parser.add_argument(
        "--dataset",
        choices=[*DATASET_CONFIGS, "both"],
        default="aldp",
        help="Evaluate outputs for aldp, aldp_full, or both (default: aldp).",
    )
    parser.add_argument(
        "--estimator-weights",
        nargs="+",
        help="Override estimator-weight values for all selected sources.",
    )
    parser.add_argument(
        "--time-stds",
        nargs="+",
        help="Override time-std values for all selected sources.",
    )
    source_flags = parser.add_argument_group("single-source shortcuts")
    source_flags.add_argument(
        "--stnce",
        action="append_const",
        const="stnce",
        dest="source_flags",
        help="Use SAVED_OUTPUT_ROOT/aldp_stnce_{i}_{j} directories (default).",
    )
    source_flags.add_argument(
        "--tnce",
        action="append_const",
        const="tnce",
        dest="source_flags",
        help="Use SAVED_OUTPUT_ROOT/aldp_tnce_{i}_{j} directories.",
    )
    source_flags.add_argument(
        "--stnce-alone",
        action="append_const",
        const="stnce_alone",
        dest="source_flags",
        help="Use SAVED_OUTPUT_ROOT/aldp_stnce_alone_{i}_{j} directories.",
    )
    source_flags.add_argument(
        "--tnce-alone",
        action="append_const",
        const="tnce_alone",
        dest="source_flags",
        help="Use SAVED_OUTPUT_ROOT/aldp_tnce_alone_{i}_{j} directories.",
    )
    source_flags.add_argument(
        "--tcnce-alone",
        action="append_const",
        const="tcnce_alone",
        dest="source_flags",
        help="Use SAVED_OUTPUT_ROOT/aldp_tcnce_alone_{i}_{j} directories.",
    )
    parser.set_defaults(source_flags=[])
    return parser.parse_args()


iid_path = "out/aldp_iid_samples.npy"
langevin_path = "out/aldp_langevin_trajectories.npy"
bins = 60
bw_method = 0.05
num_seeds = 3


def load_dataset(dataset_name: str) -> ALDPDataset:
    return ALDPDataset(
        coarse_graining_level=DATASET_CONFIGS[dataset_name],
        limit_samples=50_000,
        validation=False,
        name=dataset_name,
    )


def compute_metrics(samples, dataset, target_phi, target_psi):
    metrics = {}
    # Compute phi/psi metrics
    for name in samples.keys():
        phi_model, psi_model = dataset.get_2d_features(samples[name])
        rms_fe_sq_error, rms_mjs_error = phi_psi_metrics(
            target_phi, target_psi, phi_model, psi_model
        )
        js_div = js_divergence(
            jnp.concatenate([target_phi[None], target_psi[None]]).T,
            jnp.concatenate([phi_model[None], psi_model[None]]).T,
            bins=64,
        )
        metrics[name] = {
            "fe_sq": rms_fe_sq_error,
            "mjs": rms_mjs_error,
            "js": js_div,
            "bonds": {},
        }

    # Compute bond metrics
    for a0, a1 in dataset.bonds:
        a0_name, a1_name = dataset.atom_names[a0], dataset.atom_names[a1]
        ground_truth_bonds = pairwise_distances(dataset.train.data)[:, a0, a1]

        for name in samples.keys():
            current_bonds = pairwise_distances(samples[name])[:, a0, a1]
            w1 = wasserstein_distance(ground_truth_bonds, current_bonds)
            metrics[name]["bonds"][f"{a0_name}-{a1_name}"] = w1

    return metrics


def print_metrics(metrics_list, dataset):
    # Get all model names
    model_names = list(metrics_list[0].keys())

    # Print phi/psi metrics
    for name in model_names:
        fe_sq_values = [m[name]["fe_sq"] for m in metrics_list]
        js_values = [m[name]["js"] for m in metrics_list]
        print(f"{name}:")
        print(f"  PMF: {np.mean(fe_sq_values):.3f} ± {np.std(fe_sq_values):.3f}")
        print(f"  JS: {np.mean(js_values):.4f} ± {np.std(js_values):.4f}")

    # Print bond metrics
    for a0, a1 in dataset.bonds:
        if "Both" in model_names:
            a0_name, a1_name = dataset.atom_names[a0], dataset.atom_names[a1]
            bond_name = f"{a0_name}-{a1_name}"
            print(f"\nBond {bond_name}:")
            bond_values_best = np.array(
                [m["Both"]["bonds"][bond_name] for m in metrics_list]
            )
            for name in model_names:
                bond_values = np.array(
                    [m[name]["bonds"][bond_name] for m in metrics_list]
                )
                print(
                    f"  {name}: {np.mean(bond_values):.4f} ± {np.std(bond_values):.4f}, relative; {np.mean(bond_values / bond_values_best):.2f} ± {np.std(bond_values / bond_values_best):.2f}"
                )


if __name__ == "__main__":
    args = parse_args()
    sources = resolve_sources(args)

    for dataset_name in resolve_datasets(args):
        dataset = load_dataset(dataset_name)
        target_phi, target_psi = dataset.get_2d_features(dataset.train.data)
        data_sources = build_data_sources(
            sources,
            dataset_name,
            estimator_weights=args.estimator_weights,
            time_stds=args.time_stds,
        )

        iid_metrics_list = []
        langevin_metrics_list = []

        for seed in range(num_seeds):
            iid_samples = {}
            langevin_samples = {}

            for name, data_info in tqdm(
                data_sources.items(), desc=f"Loading {dataset_name} data for seed {seed}"
            ):
                seed_path = os.path.join(data_info.path, str(seed))
                data = onp.load(os.path.join(seed_path, iid_path))
                iid_samples[name] = data
                num_iid_samples = data.shape[0]

                data = onp.load(os.path.join(seed_path, langevin_path))
                data = data.reshape(-1, *dataset.sample_shape)
                data = jax.random.permutation(jax.random.PRNGKey(0), data)[:num_iid_samples]
                langevin_samples[name] = data

            # Compute metrics for this seed
            iid_metrics_list.append(compute_metrics(iid_samples, dataset, target_phi, target_psi))
            langevin_metrics_list.append(compute_metrics(langevin_samples, dataset, target_phi, target_psi))

        # Print results
        print(f"\nDataset: {dataset_name}")
        print("IID")
        print_metrics(iid_metrics_list, dataset)
        print("Langevin")
        print_metrics(langevin_metrics_list, dataset)
