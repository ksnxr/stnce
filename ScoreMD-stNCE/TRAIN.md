# Training the STNCE molecular experiments

The commands below replace the historical upstream mixture/Fokker–Planck recipes. For inference and final result tables, see [INFERENCE.md](INFERENCE.md).

## Running jobs

| Switch | Objective | Loss overrides |
| --- | --- | --- |
| `--run_tnce` | tNCE+DSM | `tnce_weight=1.0`, `tnce_time_std=<sweep>` |
| `--run_stnce` | stNCE-s+DSM | `stnce_weight=1.0`, `stnce_time_std=<sweep>` |
| `--run_tnce_alone` | tNCE | tNCE overrides plus `without_dsm_loss=True` |
| `--run_stnce_alone` | stNCE-s | stNCE-s overrides plus `without_dsm_loss=True` |
| `--run_tcnce_alone` | tCNCE | `tcnce_weight=1.0`, `cnce_noise_std=<sweep>`, `without_dsm_loss=True` |

Use the override table above to obtain the other objectives, and change the run name, project and output path to match.

### Coarse ALDP

```bash
srun python train.py dataset=aldp \
dataset.limit_samples=50_000 \
dataset.validation=False \
+architecture=transformer/potential \
training_schedule.epochs.0=10000 \
training_schedule.BS=1024 \
checkpoint_options.save_interval_steps=1000 \
training_schedule.losses.0.loss.beta=0.0 \
training_schedule.losses.0.loss.alpha=0.0 \
training_schedule.losses.0.loss.tnce_weight=1.0 \
training_schedule.losses.0.loss.tnce_time_std=0.01 \
+training_schedule/augment=random_rotations \
dataset.coarse_graining_level=full \
evaluation.num_parallel_langevin_samples=100 \
evaluation.langevin_dt=2e-3 \
evaluation.num_langevin_intermediate_steps=50 \
evaluation.num_langevin_samples=120000 \
evaluation.eval_t=1e-5 \
training_schedule.losses.0.loss.div_est=2 \
+training_schedule.losses.0.loss.residual_fp=True \
training_schedule.losses.0.loss.partial_t_approx=True \
+training_schedule.losses.0.loss.single_gamma=True \
wandb.enabled=True \
wandb.project=scoremd-stnce-aldp-dsm \
+wandb.name=aldp_tnce_1.0_0.01 \
hydra.run.dir=../outputs/aldp_tnce_1.0_0.01
```

### All-atom ALDP (`aldp_full`)

```bash
srun python train.py dataset=aldp_full \
dataset.limit_samples=50_000 \
dataset.validation=False \
+architecture=transformer/potential \
training_schedule.epochs.0=10000 \
training_schedule.BS=1024 \
checkpoint_options.save_interval_steps=1000 \
training_schedule.losses.0.loss.beta=0.0 \
training_schedule.losses.0.loss.alpha=0.0 \
training_schedule.losses.0.loss.tnce_weight=1.0 \
training_schedule.losses.0.loss.tnce_time_std=0.01 \
+training_schedule/augment=random_rotations \
evaluation.num_parallel_langevin_samples=100 \
evaluation.langevin_dt=2e-3 \
evaluation.num_langevin_intermediate_steps=50 \
evaluation.num_langevin_samples=120000 \
evaluation.eval_t=1e-5 \
evaluation.only_store_results=True \
training_schedule.losses.0.loss.div_est=2 \
+training_schedule.losses.0.loss.residual_fp=True \
training_schedule.losses.0.loss.partial_t_approx=True \
+training_schedule.losses.0.loss.single_gamma=True \
wandb.enabled=True \
wandb.project=scoremd-stnce-aldpfull-dsm \
+wandb.name=aldp_full_tnce_1.0_0.01 \
hydra.run.dir=../outputs/aldp_full_tnce_1.0_0.01
```

### Chignolin

```bash
srun python train.py dataset=chignolin \
+architecture=transformer/large_potential \
training_schedule.epochs=2000 \
optimizer.learning_rate=0.0001 \
training_schedule.BS=1024 \
checkpoint_options.save_interval_steps=10 \
+training_schedule/augment=random_rotations \
evaluation.num_parallel_langevin_samples=100 \
evaluation.num_langevin_samples=1000000 \
evaluation.num_iid_samples=427794 \
evaluation.inference_bs=1024 \
evaluation.fp_inference_bs=256 \
training_schedule.losses.0.loss.beta=0.0 \
training_schedule.losses.0.loss.alpha=0.0 \
training_schedule.losses.0.loss.tnce_weight=1.0 \
training_schedule.losses.0.loss.tnce_time_std=0.01 \
training_schedule.losses.0.loss.without_dsm_loss=False \
training_schedule.losses.0.loss.div_est=2 \
+training_schedule.losses.0.loss.residual_fp=True \
training_schedule.losses.0.loss.partial_t_approx=True \
+training_schedule.losses.0.loss.single_gamma=True \
evaluation.eval_t=1e-5 \
wandb.enabled=True \
wandb.project=scoremd-stnce-chignolin-dsm \
+wandb.name=chignolin_tnce_1.0_0.01 \
hydra.run.dir=../outputs/chignolin_tnce_1.0_0.01
```

## Benchmark training runtime

[`scripts/benchmark_training_runtime.py`](scripts/benchmark_training_runtime.py) is inside this repository (unlike the parent workspace's job generator).

```bash
python scripts/benchmark_training_runtime.py --dataset aldp
python scripts/benchmark_training_runtime.py --dataset chignolin
```
