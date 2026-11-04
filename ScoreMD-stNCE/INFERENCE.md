# Inference and final molecular result tables

Train the models using [TRAIN.md](TRAIN.md), then generate independent samples using diffusion sampling (IID) and Langevin trajectories for each trained configuration and inference seed. The final tables below compare those saved samples against the reference data.

### Exact generated inference command

This is the command generated for coarse ALDP, TNCE+DSM, width `0.01`, seed `0`:

```bash
srun python train.py \
  --config-path ../outputs/aldp_tnce_1.0_0.01/.hydra \
  --config-name config.yaml \
  load_dir=../outputs/aldp_tnce_1.0_0.01 \
  continue_from=null \
  seed=0 \
  evaluation.num_parallel_langevin_samples=100 \
  evaluation.langevin_dt=2e-3 \
  evaluation.num_langevin_intermediate_steps=50 \
  evaluation.num_langevin_samples=120000 \
  evaluation.num_iid_samples=120000 \
  evaluation.eval_t=1e-5 \
  evaluation.seed=0 \
  evaluation.only_store_results=False \
  wandb.enabled=True \
  wandb.project=scoremd-stnce-aldp-dsm-inference \
  wandb.name=aldp_tnce_1.0_0.01_inference_0 \
  hydra.run.dir=../outputs/aldp_tnce_1.0_0.01/0
```

For `aldp_full`, replace `aldp_` in the run paths/names with `aldp_full_`. For Chignolin, replace it with `chignolin_`. For standalone objectives, include `_alone` in the run name and omit `-dsm` from the project. For seeds 1 and 2, change both seed settings, the `_inference_<seed>` name suffix and the final output-directory component. Architecture, loss and dataset settings are restored from each training run's saved Hydra config.

## ALDP tables: coarse and all-atom

```bash
for dataset in aldp aldp_full; do
  python evaluation/evaluate_aldp_tables.py \
    --dataset "$dataset" \
    --sources tnce tcnce_alone stnce tnce_alone stnce_alone \
    --estimator-weights 1.0 \
    > "evaluation/tables/${dataset}.raw.txt"
done
```

## Chignolin tables

Use [`evaluation/evaluate_protein.py`](evaluation/evaluate_protein.py) from the repository root.
