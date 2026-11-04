# STNCE PyTorch

This package contains the PyTorch experiments for Spatial-Temporal Noise-Contrastive Estimation. The code structure follows [score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch): configs live under `configs/`, `main.py` is the train/eval entry point, and the estimator logic is split across `estimator_utils.py`, `noises_lib.py`, `losses_lib.py`, and the `*_run_lib.py` files.

First set up the environment using requirements.txt.


## Entry Point

Use `main.py` with an `ml_collections` config file:

```bash
python main.py --config configs/tele_configs.py --mode train --workdir ../logs/example
python main.py --config configs/tele_configs.py --mode eval --workdir ../logs/example
```

All config values can be overridden from the command line with
`--config.<path>=<value>`. For example:

```bash
python main.py \
  --config configs/tele_configs.py \
  --mode train \
  --config.data.name=random_gmm \
  --config.data.dim=10 \
  --config.estim.name=tstnce \
  --config.optim.lr=0.0001 \
  --workdir ../logs/random_gmm_tstnce
```

The available config files are:

- `configs/simple_tele_configs.py`: smaller synthetic defaults.
- `configs/tele_configs.py`: synthetic experiments.
- `configs/image_configs.py`: MNIST mixture image defaults.
- `configs/mnist_configs.py`: MNIST image experiments.
- `configs/imagenet_configs.py`: ImageNet64 experiments.

## Data Setup

`random_gmm` as implemented in `data/toy_data.py` and `mnist_mixture` as implemented in `data/data.py` were used in the paper.

For `random_gmm` and `mnist_mixture`, generate the dataset itself by setting `training.num_iters=0` and `training.setup=True`:

```bash
python main.py \
  --config configs/tele_configs.py \
  --mode train \
  --config.training.setup=True \
  --config.training.num_iters=0 \
  --config.data.name=random_gmm \
  --config.data.dim=10 \
  --config.data.num_components=20 \
  --config.data.sigma=0.1 \
  --config.data.mu_scale=1.0 \
  --config.seed=1
```

and generate cached train/val/test/importance-sampling sets
before training by setting `training.num_iters=0`:

```bash
python main.py \
  --config configs/tele_configs.py \
  --mode train \
  --config.training.num_iters=0 \
  --config.data.name=random_gmm \
  --config.data.dim=10 \
  --config.data.num_components=20 \
  --config.data.sigma=0.1 \
  --config.data.mu_scale=1.0 \
  --config.seed=1
```

Image datasets are handled by `data/data.py`. MNIST is downloaded through
`torchvision`. ImageNet64 expects data under `../../imagenet64` relative to this
directory.

## Estimators

Set the primary estimator with `--config.estim.name=...`.

Common choices:

- `tnce`,
- `tcnce`,
- `tstnce`: STNCE,
- `ctsm`,
- `dsm`.

Regularizers are controlled with `--config.estim.reg_estimator=...`. Supported
regularizers are `none`, `tcnce`, `dsm`, and `fm`, depending on the primary estimator.

## Common Training Commands

### STNCE-self With Reuse

```bash
python main.py \
  --config configs/tele_configs.py \
  --mode train \
  --config.data.name=random_gmm \
  --config.data.dim=10 \
  --config.data.num_components=20 \
  --config.data.sigma=0.1 \
  --config.data.mu_scale=1.0 \
  --config.model.name=small_resnet \
  --config.estim.prob_path=ot \
  --config.estim.name=tstnce \
  --config.estim.combine_type=reuse \
  --config.estim.time_std=-1.0 \
  --config.estim.stnce_noise_type=diff \
  --config.estim.score_type=self \
  --config.estim.reg_estimator=none \
  --config.estim.resolve_collisions=True \
  --config.optim.lr=0.0001 \
  --config.training.batch_size=125 \
  --config.eval.batch_size=250 \
  --config.seed=1 \
  --config.print_test=True \
  --workdir ../logs/random_gmm_tstnce_reuse_self
```

### TNCE with reuse

```bash
python main.py \
  --config configs/tele_configs.py \
  --mode train \
  --config.data.name=random_gmm \
  --config.data.dim=10 \
  --config.model.name=small_resnet \
  --config.estim.name=tnce \
  --config.estim.combine_type=reuse \
  --config.estim.score_type=none \
  --config.estim.resolve_collisions=True \
  --config.estim.time_std=0.1 \
  --config.estim.reg_estimator=none \
  --config.optim.lr=0.0003 \
  --workdir ../logs/random_gmm_tnce
```

### TNCE + TCNCE Mixture with reuse

Referred to as STNCE-mixture.

```bash
python main.py \
  --config configs/tele_configs.py \
  --mode train \
  --config.data.name=random_gmm \
  --config.data.dim=10 \
  --config.model.name=small_resnet \
  --config.estim.name=tnce \
  --config.estim.reg_estimator=tcnce \
  --config.estim.combine_type=reuse \
  --config.estim.time_std=-1.0 \
  --config.estim.cnce_noise_type=white-0.1 \
  --config.estim.mixture_loss=True \
  --config.estim.resolve_collisions=True \
  --config.optim.lr=0.0001 \
  --workdir ../logs/random_gmm_tnce_tcnce_reuse
```

### DSM

```bash
python main.py \
  --config configs/tele_configs.py \
  --mode train \
  --config.data.name=random_gmm \
  --config.data.dim=10 \
  --config.model.name=small_resnet \
  --config.estim.name=dsm \
  --config.estim.reg_estimator=none \
  --config.estim.main_weight=1.0 \
  --config.optim.lr=0.0003 \
  --workdir ../logs/random_gmm_dsm
```

### Dual SM (CTSM + DSM)

```bash
python main.py \
  --config configs/tele_configs.py \
  --mode train \
  --config.data.name=random_gmm \
  --config.data.dim=10 \
  --config.model.name=small_resnet \
  --config.estim.name=ctsm \
  --config.estim.reg_estimator=dsm \
  --config.estim.combine_type=default \
  --config.estim.score_type=none \
  --config.estim.time_score_type=none \
  --config.estim.t_dist=uniform \
  --config.optim.lr=0.0003 \
  --workdir ../logs/random_gmm_ctsm_dsm
```

### RNE

RNE is generated using `--config.estim.name=tstnce` with RNE-specific switches: `--config.estim.rne=True`, `--config.estim.stnce_noise_type=inf-diff`, and usually `--config.estim.rne_reference=True`. `time_std` is between `0.00001`, `0.0001`, and `0.001`, and uses `main_weight=1000`.

RNE-self without a regularizer:

```bash
python main.py \
  --config configs/tele_configs.py \
  --mode train \
  --config.data.name=random_gmm \
  --config.data.dim=10 \
  --config.model.name=small_resnet \
  --config.estim.name=tstnce \
  --config.estim.rne=True \
  --config.estim.stnce_noise_type=inf-diff \
  --config.estim.combine_type=default \
  --config.estim.score_type=self \
  --config.estim.time_std=0.0001 \
  --config.estim.main_weight=1000 \
  --config.estim.rne_reference=True \
  --config.estim.reg_estimator=none \
  --config.optim.lr=0.0003 \
  --workdir ../logs/random_gmm_rne_self
```

RNE-self with DSM regularization:

```bash
python main.py \
  --config configs/tele_configs.py \
  --mode train \
  --config.data.name=random_gmm \
  --config.data.dim=10 \
  --config.model.name=small_resnet \
  --config.estim.name=tstnce \
  --config.estim.rne=True \
  --config.estim.stnce_noise_type=inf-diff \
  --config.estim.combine_type=default \
  --config.estim.score_type=self \
  --config.estim.time_std=0.0001 \
  --config.estim.main_weight=1000 \
  --config.estim.rne_reference=True \
  --config.estim.reg_estimator=dsm \
  --config.optim.lr=0.0003 \
  --workdir ../logs/random_gmm_rne_self_dsm
```

Use `score_type=oracle` for the oracle variants.

Example job generation:

```bash
cd ..
python scripts/tool.py \
  --mode init \
  --datas random_gmm \
  --dims 10 \
  --models small_resnet \
  --prob_paths ot \
  --run_rne_default_self \
  --lrs 0.0003
```

### Image

```bash
python main.py \
  --config configs/mnist_configs.py \
  --mode train \
  --config.data.name=mnist \
  --config.estim.name=tstnce \
  --config.estim.combine_type=reuse \
  --config.estim.stnce_noise_type=diff \
  --config.estim.score_type=self \
  --config.estim.reg_estimator=none \
  --config.estim.resolve_collisions=True \
  --config.optim.lr=0.0001 \
  --workdir ../logs/mnist_tstnce_reuse_self \
  --config.data.use_dequantization=True
```

For image training, the run libraries use `accelerate`. Multi-process and mixed precision runs can be launched with `accelerate launch`, for example:

```bash
accelerate launch --mixed_precision bf16 main.py \
  --config configs/imagenet_configs.py \
  --mode train \
  --config.data.name=imagenet64 \
  --workdir ../logs/imagenet64_run \
  --config.data.use_dequantization=False
```

## Evaluation

Evaluate a trained run by pointing `--workdir` at the training directory:

```bash
python main.py \
  --config configs/mnist_configs.py \
  --mode eval \
  --workdir ../logs/mnist_tstnce_reuse_self
  --config.eval.checkpoint_name=...
```

## MNIST-mixture error analysis

After training models on MNIST-mixture, run the following to perform a fine-grained analysis.

```bash
python -m analysis.analyze_local_to_global_ratio_errors \
  --output-dir ../logs/mnist_mixture_test/local_to_global_error_analysis
```
