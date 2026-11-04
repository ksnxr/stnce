#!/usr/bin/env python3
"""Benchmark DSM, TNCE, TNCE+DSM, STNCE, STNCE+DSM, and FPE for molecules.

Run the dataset shell wrappers from the repository root in an activated Python
environment, or invoke this runner with --dataset aldp or chignolin.
Configuration is hardcoded in DATASET_SETTINGS and main(). Set dry=True to
write commands without training, or full_run=True for production-length training.
"""

import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time


DATASET_SETTINGS = {
    "aldp": {
        "architecture": "transformer/potential",
        "full_epochs": 10000,
        "learning_rate": "0.0003",
        "fpe_beta": 0.0005,
        "time_stds": {"tnce": 0.1, "tnce_dsm": 0.01, "stnce": 0.1, "stnce_dsm": 0.1},
    },
    "chignolin": {
        "architecture": "transformer/large_potential",
        "full_epochs": 2000,
        "learning_rate": "0.0001",
        "fpe_beta": 0.002,
        "time_stds": {"tnce": 0.0, "tnce_dsm": 0.1, "stnce": 0.1, "stnce_dsm": 0.01},
    },
}


def measured_training_seconds(text, warmup, total):
    """Sum synchronized training calls after warmup, rejecting incomplete logs."""
    matches = re.findall(r"Training compute epoch (\d+): ([0-9.eE+-]+) seconds", text)
    timings = {}
    for epoch, seconds in matches:
        epoch, seconds = int(epoch), float(seconds)
        if warmup < epoch <= total:
            if epoch in timings or not math.isfinite(seconds) or seconds <= 0:
                raise ValueError(f"Invalid or duplicate training timing for epoch {epoch}")
            timings[epoch] = seconds
    if set(timings) != set(range(warmup + 1, total + 1)):
        raise ValueError("Missing measured training epochs")
    return sum(timings.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASET_SETTINGS)
    args = parser.parse_args()
    dataset = args.dataset
    settings = DATASET_SETTINGS[dataset]
    bench = 10
    warmup = 2
    full = settings["full_epochs"]
    bs = 1024
    devices = 1
    if bs % devices:
        raise SystemExit("BS must be divisible by NUM_DEVICES")
    time_stds = dict(settings["time_stds"])
    # Standalone FPE settings from noegroup/ScoreMD TRAIN.md.
    beta = settings["fpe_beta"]
    if beta <= 1e-6:
        raise SystemExit("FPE_BETA must exceed 1e-6 so the FP loss is enabled")
    seed = 1
    full_run = False
    total = full if full_run else warmup + bench
    if total <= warmup:
        raise SystemExit("Total epochs must exceed WARMUP_EPOCHS")
    bench = total - warmup
    skip_evaluation = True
    wandb_enabled = "False"
    if wandb_enabled not in ("True", "False"):
        raise SystemExit("WANDB_ENABLED must be True or False")
    dry = False
    work = Path(__file__).resolve().parents[1]
    os.chdir(work)
    sys.path.insert(0, str(work / "src"))
    # Direct Python invocation must expose the source tree to training subprocesses.
    os.environ["PYTHONPATH"] = str(work / "src")
    os.environ["MPLBACKEND"] = "Agg"
    root = work / "outputs" / dataset / "bench_training_runtime" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root.mkdir(parents=True, exist_ok=False)


    def log(message):
        print(message, flush=True)
        with (root / "summary.txt").open("a") as stream:
            stream.write(message + "\n")


    log(f"{dataset}: six-method runtime benchmark; results: {root}")
    log(f"Each method: {warmup} warmup + {bench} measured epochs; projection: {full} epochs")
    log(f"FPE means DSM + FP penalty (beta={beta}). Per-method temporal noise: {time_stds}")
    log(f"Evaluation enabled: {not skip_evaluation}; W&B enabled: {wandb_enabled}")
    log("Validation and regular checkpoint intervals: 1000000 epochs.")
    log("Projection sums synchronized training calls after warmup; excludes checkpointing, validation, and evaluation.")
    if not dry:
        import jax
        if jax.default_backend() != "gpu" or jax.local_device_count() < devices:
            raise SystemExit(f"Need {devices} local JAX GPU device(s); found {jax.devices()}")
        (root / "environment.txt").write_text(
            f"Python: {sys.version}\nJAX: {jax.__version__}\nDevices: {jax.devices()}\n")
        with (root / "packages.txt").open("w") as stream:
            from importlib.metadata import distributions
            for dist in sorted(distributions(), key=lambda d: d.metadata.get("Name", "")):
                stream.write(f"{dist.metadata.get('Name')}=={dist.version}\n")

    common = [
        f"dataset={dataset}", f"+architecture={settings['architecture']}",
        f"optimizer.learning_rate={settings['learning_rate']}", f"training_schedule.BS={bs}",
        f"training_schedule.epochs=[{total}]", "training_schedule.dsm_warmup_epochs=0",
        "training_schedule.benchmark_timing=True",
        "+training_schedule/augment=random_rotations",
        "checkpoint_options.save_interval_steps=1000000",
        "training_schedule.validation_every=1000000",
        f"num_devices={devices}", f"seed={seed}",
        f"no_evaluation={skip_evaluation}", f"wandb.enabled={wandb_enabled}",
        "training_schedule.losses.0.loss.alpha=0.0",
        "training_schedule.losses.0.loss.diffclf_weight=0.0",
        "training_schedule.losses.0.loss.tcnce_weight=0.0",
        "training_schedule.losses.0.loss.rne_weight=0.0",
        "training_schedule.losses.0.loss.div_est=2",
        "+training_schedule.losses.0.loss.residual_fp=True",
        "training_schedule.losses.0.loss.partial_t_approx=True",
        "+training_schedule.losses.0.loss.single_gamma=True",
    ]
    common += ["evaluation.num_parallel_langevin_samples=100", "evaluation.eval_t=1e-5"]
    if dataset == "aldp":
        aldp_samples = 50000
        common += [f"dataset.limit_samples={aldp_samples}",
                   "dataset.validation=False", "evaluation.langevin_dt=2e-3",
                   "evaluation.num_langevin_intermediate_steps=50",
                   "evaluation.num_langevin_samples=120000"]
        common += ["dataset.coarse_graining_level=full"]
    else:
        common += ["evaluation.num_langevin_samples=1000000",
                   "evaluation.num_iid_samples=427794", "evaluation.inference_bs=1024",
                   "evaluation.fp_inference_bs=256"]

    # label, TNCE weight, STNCE weight, omit DSM, FP weight
    methods = [
        ("dsm", 0, 0, False, 0),
        ("tnce", 1, 0, True, 0),
        ("tnce_dsm", 1, 0, False, 0),
        ("stnce", 0, 1, True, 0),
        ("stnce_dsm", 0, 1, False, 0),
        ("fpe", 0, 0, False, beta),
    ]
    fields = ["method", "time_std", "status", "warmup_epochs", "measured_epochs", "process_wall_s",
              "measured_interval_s", "seconds_per_epoch", "full_epochs", "projected_training_h"]
    with (root / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        stream.flush()
        runs = [(method, time_stds.get(method[0])) for method in methods]
        for (method, tnce, stnce, without_dsm, fp), time_std in runs:
            label = method if time_std is None else f"{method}_time_std_{time_std}"
            project = f"scoremd-stnce-{dataset}" + ("-dsm" if not without_dsm else "")
            noise_args = []
            if tnce != 0:
                noise_args = [f"training_schedule.losses.0.loss.tnce_time_std={time_std}"]
            elif stnce != 0:
                noise_args = [f"training_schedule.losses.0.loss.stnce_time_std={time_std}"]
            directory = root / label
            directory.mkdir()
            command = ["srun", sys.executable, "-u", "train.py", *common, *noise_args,
                f"wandb.project={project}", f"+wandb.name=benchmark_{dataset}_{label}",
                f"training_schedule.losses.0.loss.tnce_weight={tnce}",
                f"training_schedule.losses.0.loss.stnce_weight={stnce}",
                f"training_schedule.losses.0.loss.without_dsm_loss={without_dsm}",
                f"training_schedule.losses.0.loss.beta={fp}",
                f"hydra.run.dir={directory}"]
            (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
            log(f"\n{label}: {shlex.join(command)}")
            if dry:
                continue
            row = dict(method=method, time_std=time_std, status="failed", warmup_epochs=warmup,
                       measured_epochs=bench, full_epochs=full)
            start = time.perf_counter()
            with (directory / "train.log").open("w") as output:
                result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
            row["process_wall_s"] = round(time.perf_counter() - start, 6)
            text = (directory / "train.log").read_text(errors="replace")
            if result.returncode:
                writer.writerow(row)
                stream.flush()
                raise SystemExit(f"{label} failed. See {directory / 'train.log'}")
            try:
                elapsed = measured_training_seconds(text, warmup, total)
                if "Retrying epoch" in text or "marking this epoch as non-special" in text:
                    raise ValueError("Epoch retries detected")
            except ValueError as error:
                row["status"] = "invalid_timing_or_retries"
                writer.writerow(row)
                stream.flush()
                raise SystemExit(f"{label}: {error}; inspect {directory / 'train.log'}") from error
            per_epoch = elapsed / bench
            row.update(status="ok", measured_interval_s=round(elapsed, 6),
                       seconds_per_epoch=round(per_epoch, 6),
                       projected_training_h=round(per_epoch * full / 3600, 6))
            writer.writerow(row)
            stream.flush()
            log(f"{label}: {per_epoch:.3f} s/epoch; projected {full} epochs: "
                f"{per_epoch * full / 3600:.3f} h; process wall: {row['process_wall_s']:.3f} s")
    log("Dry run complete." if dry else "All benchmark configurations completed.")


if __name__ == "__main__":
    main()
