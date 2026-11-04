#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np

# Ensure we can import the ScoreMD package (src layout)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(THIS_DIR, "src")
if os.path.isdir(SRC_DIR) and SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from scoremd.data.dataset.aldp import ALDPDataset  # type: ignore
import jax.numpy as jnp  # type: ignore
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore


def main():
    parser = argparse.ArgumentParser(
        description="Plot ALDP Ramachandran-like (phi/psi) histograms for GT, IID, and Langevin from existing outputs."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Base directory containing 'out' with data files.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for plots. Defaults to '<base-dir>/out'.",
    )
    parser.add_argument(
        "--iid-npy",
        type=str,
        default=None,
        help="Path to IID samples .npy. Defaults to '<out-dir>/aldp_iid_samples.npy'.",
    )
    parser.add_argument(
        "--langevin-trajectories-npy",
        type=str,
        default=None,
        help="Path to Langevin trajectories .npy. Defaults to '<out-dir>/aldp_langevin_trajectories.npy'.",
    )
    args = parser.parse_args()

    base_dir = os.path.expanduser(args.base_dir)
    out_dir = os.path.expanduser(args.out_dir) if args.out_dir else os.path.join(base_dir, "out")
    os.makedirs(out_dir, exist_ok=True)

    iid_npy = os.path.expanduser(args.iid_npy) if args.iid_npy else os.path.join(out_dir, "aldp_iid_samples.npy")
    langevin_traj_npy = (
        os.path.expanduser(args.langevin_trajectories_npy)
        if args.langevin_trajectories_npy
        else os.path.join(out_dir, "aldp_langevin_trajectories.npy")
    )

    # ALDP dataset (phi/psi plotting via ALDPDataset)
    dataset = ALDPDataset(validation=False)

    # 1) Ground truth histogram (dataset.train.data)
    plt.figure(clear=True)
    dataset.plot_2d(dataset.train.data, title=None)
    gt_png = os.path.join(out_dir, "aldp_phi_psi_ground_truth.png")
    plt.savefig(gt_png, bbox_inches="tight")
    plt.close()
    print(f"[OK] Wrote {gt_png}")

    # 2) IID samples histogram
    if not os.path.isfile(iid_npy):
        raise FileNotFoundError(f"IID samples not found: {iid_npy}")
    iid_samples = np.load(iid_npy)
    plt.figure(clear=True)
    dataset.plot_2d(jnp.array(iid_samples), title=None)
    iid_png = os.path.join(out_dir, "aldp_phi_psi_iid.png")
    plt.savefig(iid_png, bbox_inches="tight")
    plt.close()
    print(f"[OK] Wrote {iid_png}")

    # 3) Langevin samples histogram (flatten trajectories over all points and steps)
    if not os.path.isfile(langevin_traj_npy):
        raise FileNotFoundError(f"Langevin trajectories not found: {langevin_traj_npy}")
    traj = np.load(langevin_traj_npy)  # shape: (n_points, n_steps, -1)
    traj_flat = traj.reshape(-1, traj.shape[-1])
    plt.figure(clear=True)
    dataset.plot_2d(jnp.array(traj_flat), title=None)
    langevin_png = os.path.join(out_dir, "aldp_phi_psi_langevin.png")
    plt.savefig(langevin_png, bbox_inches="tight")
    plt.close()
    print(f"[OK] Wrote {langevin_png}")


if __name__ == "__main__":
    main()
