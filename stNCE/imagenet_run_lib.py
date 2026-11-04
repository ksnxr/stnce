import os
import math
import time
import random
import torch
import logging
import losses_lib
import pickle
import numpy as np
from accelerate import Accelerator, DataLoaderConfiguration
from torch.optim import lr_scheduler

from models.ema import ExponentialMovingAverage
import wandb

# for registrations
from models import unet

from models import model_utils as mutils
from data.data import create_datasets
from absl import flags
from estimator_utils import build_estimator_fns
from utils import (
    seed_all,
    seed_rank_rng,
    scaler,
    scaler_data_statistics,
    image_evaluate,
    inverse_scaler,
    VelocityWrapper,
    euler_sample,
)

from torchvision.utils import save_image
from prob_paths_lib import get_prob_path


FLAGS = flags.FLAGS


def _synchronized_perf_counter(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _unwrap_model_for_log_const(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def _get_precond_specified_log_const(model, config, device):
    if not config.model.name.startswith("precond_"):
        return None

    model = _unwrap_model_for_log_const(model)
    base_log_const = 0.5 * math.log(2.0 * math.pi) * config.data.dim
    if (not hasattr(model, "log_normalizer")) or (
        not getattr(model, "include_time_log_normalizer_in_output", False)
    ):
        return float(base_log_const)

    noise_t = torch.full(
        (1,), fill_value=config.estim.eps1, device=device, dtype=torch.float32
    )
    log_normalizer_t0 = model.log_normalizer(noise_t).reshape(()).item()
    return float(base_log_const + log_normalizer_t0)


def _build_precond_specified_log_const(config, device):
    if not config.model.name.startswith("precond_"):
        return None

    def current_log_const(model):
        return _get_precond_specified_log_const(model, config, device)

    return current_log_const


def _get_scheduler_config(config):
    return {
        "name": config.scheduler.name,
        "warmup_steps": int(config.scheduler.warmup_steps),
        "decay_start_step": int(config.scheduler.decay_start_step),
        "min_lr": float(config.scheduler.min_lr),
        "min_lr_ratio": float(config.scheduler.min_lr_ratio),
        "step_size": int(config.scheduler.step_size),
        "gamma": float(config.scheduler.gamma),
    }


def _create_lr_scheduler(config, optimizer, num_train_steps):
    scheduler_config = _get_scheduler_config(config)
    scheduler_name = scheduler_config["name"]
    base_lr = float(config.optim.lr)

    if scheduler_name == "flat":
        warmup_steps = max(0, scheduler_config["warmup_steps"])
        if warmup_steps > 0:
            return lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: min(1.0, (step + 1) / warmup_steps),
            )
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    min_lr = scheduler_config["min_lr"]
    if min_lr <= 0.0:
        min_lr = base_lr * scheduler_config["min_lr_ratio"]

    if scheduler_name == "step":
        return lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, scheduler_config["step_size"]),
            gamma=scheduler_config["gamma"],
        )

    if scheduler_name == "cosine":
        decay_start_step = max(0, scheduler_config["decay_start_step"])
        if decay_start_step == 0:
            return lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, num_train_steps),
                eta_min=min_lr,
            )

        return lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                lr_scheduler.ConstantLR(
                    optimizer,
                    factor=1.0,
                    total_iters=decay_start_step,
                ),
                lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, num_train_steps - decay_start_step),
                    eta_min=min_lr,
                ),
            ],
            milestones=[decay_start_step],
        )

    if scheduler_name == "warmup_cosine":
        warmup_steps = max(0, scheduler_config["warmup_steps"])
        decay_start_step = max(warmup_steps, scheduler_config["decay_start_step"])
        schedulers = []
        phase_lengths = []

        if warmup_steps > 0:
            schedulers.append(
                lr_scheduler.LambdaLR(
                    optimizer,
                    lr_lambda=lambda step: min(1.0, (step + 1) / warmup_steps),
                )
            )
            phase_lengths.append(warmup_steps)

        if decay_start_step > warmup_steps:
            schedulers.append(
                lr_scheduler.ConstantLR(
                    optimizer,
                    factor=1.0,
                    total_iters=decay_start_step - warmup_steps,
                )
            )
            phase_lengths.append(decay_start_step - warmup_steps)

        remaining_steps = num_train_steps - decay_start_step
        if remaining_steps > 0:
            schedulers.append(
                lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, remaining_steps),
                    eta_min=min_lr,
                )
            )
            phase_lengths.append(remaining_steps)

        if not schedulers:
            return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        if len(schedulers) == 1:
            return schedulers[0]

        milestones = []
        elapsed_steps = 0
        for phase_length in phase_lengths[:-1]:
            elapsed_steps += phase_length
            milestones.append(elapsed_steps)
        return lr_scheduler.SequentialLR(
            optimizer,
            schedulers=schedulers,
            milestones=milestones,
        )

    raise ValueError(f"Unsupported learning rate scheduler: {scheduler_name}")


def _apply_config_learning_rate_to_resume(optimizer, scheduler, learning_rate):
    learning_rate = float(learning_rate)
    old_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
        if "initial_lr" in group:
            group["initial_lr"] = learning_rate

    new_lrs = [learning_rate for _ in optimizer.param_groups]
    _set_scheduler_lrs(scheduler, new_lrs)
    return old_lrs, new_lrs


def _set_scheduler_lrs(scheduler, learning_rates):
    if hasattr(scheduler, "base_lrs"):
        scheduler.base_lrs = list(learning_rates)
    if hasattr(scheduler, "_last_lr"):
        scheduler._last_lr = list(learning_rates)
    for child_scheduler in getattr(scheduler, "_schedulers", []):
        _set_scheduler_lrs(child_scheduler, learning_rates)


def _base_checkpoint_path(checkpoint_path):
    root, ext = os.path.splitext(checkpoint_path)
    if ".rank" in root:
        prefix, suffix = root.rsplit(".rank", 1)
        if suffix.isdigit():
            root = prefix
    return root, ext


def _get_process_checkpoint_path(checkpoint_path, process_index, num_processes):
    root, ext = _base_checkpoint_path(checkpoint_path)
    if num_processes == 1:
        return f"{root}{ext}"
    return f"{root}.rank{process_index}{ext}"


def _get_non_main_process_checkpoint_paths(checkpoint_path, num_processes):
    if num_processes == 1:
        return []
    return [
        _get_process_checkpoint_path(checkpoint_path, process_index, num_processes)
        for process_index in range(0, num_processes)
    ]


def _resolve_resume_checkpoint_path(checkpoint_path, process_index, num_processes):
    process_checkpoint_path = _get_process_checkpoint_path(
        checkpoint_path, process_index, num_processes
    )
    if os.path.exists(process_checkpoint_path):
        return process_checkpoint_path
    return checkpoint_path


def _resolve_checkpoint_path(checkpoints_dir, checkpoint_name):
    if os.path.isabs(checkpoint_name):
        return checkpoint_name
    return os.path.join(checkpoints_dir, checkpoint_name)


def _resolve_eval_checkpoint_path(workdir, checkpoint_name):
    checkpoints_dir = os.path.join(workdir, "checkpoints")
    checkpoint_path = _resolve_checkpoint_path(checkpoints_dir, checkpoint_name)
    if os.path.exists(checkpoint_path) or os.path.isabs(checkpoint_name):
        return checkpoint_path

    legacy_checkpoint_path = os.path.join(workdir, checkpoint_name)
    if os.path.exists(legacy_checkpoint_path):
        return legacy_checkpoint_path

    return checkpoint_path


def _get_rng_state():
    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    if torch.cuda.is_available():
        rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return rng_state


def _coerce_rng_byte_tensor(rng_state_tensor):
    if isinstance(rng_state_tensor, np.ndarray):
        rng_state_tensor = torch.from_numpy(rng_state_tensor)
    elif not isinstance(rng_state_tensor, torch.Tensor):
        rng_state_tensor = torch.as_tensor(rng_state_tensor, dtype=torch.uint8)
    return rng_state_tensor.detach().to(device="cpu", dtype=torch.uint8)


def _set_rng_state(rng_state):
    if rng_state is None:
        return

    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(_coerce_rng_byte_tensor(rng_state["torch"]))

    if torch.cuda.is_available() and ("torch_cuda" in rng_state):
        torch.cuda.set_rng_state_all(
            [_coerce_rng_byte_tensor(state) for state in rng_state["torch_cuda"]]
        )

    if "cudnn_deterministic" in rng_state:
        torch.backends.cudnn.deterministic = rng_state["cudnn_deterministic"]
    if "cudnn_benchmark" in rng_state:
        torch.backends.cudnn.benchmark = rng_state["cudnn_benchmark"]


def _get_dataloader_epoch(train_dataloader):
    for attr_name in ("iteration", "epoch"):
        epoch = getattr(train_dataloader, attr_name, None)
        if epoch is not None:
            return int(epoch)
    return 0


def _set_dataloader_epoch(train_dataloader, epoch):
    if hasattr(train_dataloader, "set_epoch"):
        train_dataloader.set_epoch(int(epoch))


def _create_train_iterator(train_dataloader):
    dataloader_epoch = _get_dataloader_epoch(train_dataloader)
    _set_dataloader_epoch(train_dataloader, dataloader_epoch)
    train_iterator_state = {
        "dataloader_epoch": dataloader_epoch,
        "epoch_start_rng_state": _get_rng_state(),
        "steps_into_epoch": 0,
    }
    return iter(train_dataloader), train_iterator_state


def _restore_train_iterator(train_dataloader, train_iterator_state):
    dataloader_epoch = int(train_iterator_state.get("dataloader_epoch", 0))
    epoch_start_rng_state = train_iterator_state.get("epoch_start_rng_state")
    steps_into_epoch = int(train_iterator_state.get("steps_into_epoch", 0))

    if epoch_start_rng_state is None:
        _set_dataloader_epoch(train_dataloader, dataloader_epoch)
        return iter(train_dataloader)

    _set_rng_state(epoch_start_rng_state)
    _set_dataloader_epoch(train_dataloader, dataloader_epoch)
    train_iter = iter(train_dataloader)
    for _ in range(steps_into_epoch):
        try:
            next(train_iter)
        except StopIteration as err:
            raise RuntimeError(
                "Checkpoint dataloader state is inconsistent with the current "
                "training dataloader."
            ) from err
    return train_iter


class _IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        if isinstance(sample, tuple):
            return (*sample, index)
        if isinstance(sample, list):
            return (*sample, index)
        return sample, index


def _add_sample_indices_to_dataloader(dataloader):
    dataloader_kwargs = {
        "batch_sampler": dataloader.batch_sampler,
        "num_workers": dataloader.num_workers,
        "collate_fn": dataloader.collate_fn,
        "pin_memory": dataloader.pin_memory,
        "timeout": dataloader.timeout,
        "worker_init_fn": dataloader.worker_init_fn,
        "multiprocessing_context": dataloader.multiprocessing_context,
    }
    if dataloader.num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = dataloader.prefetch_factor
        dataloader_kwargs["persistent_workers"] = dataloader.persistent_workers

    pin_memory_device = getattr(dataloader, "pin_memory_device", "")
    if pin_memory_device:
        dataloader_kwargs["pin_memory_device"] = pin_memory_device

    return torch.utils.data.DataLoader(
        _IndexedDataset(dataloader.dataset),
        **dataloader_kwargs,
    )


def _get_local_count(total_count, process_index, num_processes):
    total_count = int(total_count)
    if total_count < 0:
        raise ValueError(f"total_count must be non-negative, got {total_count}.")
    base_count = total_count // num_processes
    remainder = total_count % num_processes
    return base_count + int(process_index < remainder)


def _build_imagenet_data_statistics(config, device):
    return {
        "means": torch.zeros((config.data.dim,), device=device, dtype=torch.float32),
        "std": 1.0,
    }


def _gather_variable_batch_to_main(local_batch, local_count, total_count, accelerator):
    total_count = int(total_count)
    local_count_tensor = torch.tensor([local_count], device=local_batch.device)
    gathered_counts = accelerator.gather(local_count_tensor)
    max_count = int(gathered_counts.max().item())

    if max_count == 0:
        return local_batch.detach().cpu() if accelerator.is_main_process else None

    padded_batch = local_batch.new_zeros((max_count, *local_batch.shape[1:]))
    valid_mask = torch.zeros((max_count,), device=local_batch.device, dtype=torch.long)
    if local_count > 0:
        padded_batch[:local_count] = local_batch
        valid_mask[:local_count] = 1

    gathered_batch = accelerator.gather(padded_batch)
    gathered_mask = accelerator.gather(valid_mask).to(dtype=torch.bool)

    if not accelerator.is_main_process:
        return None

    gathered_batch = gathered_batch[gathered_mask]
    return gathered_batch[:total_count].detach().cpu()


def _build_training_checkpoint(
    accelerator,
    state,
    step,
    metrics,
    best_metric_values,
    train_time_between_eval_steps,
    train_iterator_state,
    start_time,
    device,
):
    checkpoint_state = {
        "step": step,
        "process_index": accelerator.process_index,
        "num_processes": accelerator.num_processes,
        "model_state_dict": accelerator.unwrap_model(state["model"]).state_dict(),
        "optimizer_state_dict": state["optimizer"].state_dict(),
        "metrics": metrics,
        "best_metric_values": best_metric_values,
        "train_time_between_eval_steps": train_time_between_eval_steps,
        "rng_state": _get_rng_state(),
        "accumulated_train_time": max(
            _synchronized_perf_counter(device) - start_time, 0.0
        ),
    }

    if train_iterator_state is not None:
        checkpoint_state["train_iterator_state"] = train_iterator_state

    if "scheduler" in state:
        checkpoint_state["scheduler_state_dict"] = state["scheduler"].state_dict()

    if "ema" in state:
        checkpoint_state["ema_state_dict"] = state["ema"].state_dict()

    grad_scaler = getattr(accelerator, "scaler", None)
    if grad_scaler is not None:
        checkpoint_state["grad_scaler_state_dict"] = grad_scaler.state_dict()

    return checkpoint_state


def train(config, workdir):
    start_time = _synchronized_perf_counter(torch.device(config.device))
    seed_all(config.seed, exact=(config.data.name != "imagenet64"))

    num_train_steps = config.training.num_iters

    mixed_precision = getattr(config.training, "mixed_precision", "no")
    if mixed_precision not in {"no", "fp16", "bf16"}:
        raise ValueError(
            "config.training.mixed_precision must be one of 'no', 'fp16', or "
            f"'bf16', got {mixed_precision!r}"
        )
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        dataloader_config=DataLoaderConfiguration(
            split_batches=config.training.split_batches,
            even_batches=config.training.even_batches,
            use_seedable_sampler=True,
        ),
    )
    device = accelerator.device
    estimator = config.estim.name
    reg_estimator = config.estim.reg_estimator

    num_noise_classes = config.estim.num_noise_classes
    use_wandb = config.wandb.use_wandb
    save_checkpoint = config.training.save_checkpoint
    save_all_checkpoints = config.training.save_all_checkpoints
    resume_checkpoint = config.training.resume_checkpoint
    latest_checkpoint_name = config.training.latest_checkpoint_name
    checkpoints_dir = os.path.join(workdir, "checkpoints")
    latest_checkpoint_path = _resolve_checkpoint_path(
        checkpoints_dir, latest_checkpoint_name
    )
    latest_process_checkpoint_path = _get_process_checkpoint_path(
        latest_checkpoint_path, accelerator.process_index, accelerator.num_processes
    )
    if resume_checkpoint != "":
        resume_checkpoint = _resolve_checkpoint_path(checkpoints_dir, resume_checkpoint)

    if accelerator.is_main_process:
        logging.info(
            "Accelerate runtime: mixed_precision=%s, device=%s, num_processes=%s",
            accelerator.mixed_precision,
            accelerator.device,
            accelerator.num_processes,
        )
        logging.info(
            "Launch training with Accelerate for mixed precision, for example: "
            "accelerate launch --mixed_precision bf16 main.py --config "
            "configs/imagenet_configs.py --mode train ..."
        )

    if save_checkpoint or save_all_checkpoints:
        if accelerator.is_main_process:
            os.makedirs(checkpoints_dir, exist_ok=True)
        accelerator.wait_for_everyone()

    # use wandb
    if use_wandb and accelerator.is_main_process:
        wandb.login()
        wandb.init(
            project=config.wandb.project_name,
            name=config.wandb.run_name,
            config=config,
            dir=workdir,
        )

    (
        train_temp,
        val_dataloader,
        test_dataloader,
        is_dataloader,
        is_infinite,
        data_dist,
        data_statistics,
    ) = create_datasets(config, device=device, process_index=accelerator.process_index)
    if is_dataloader is not None:
        raise RuntimeError("ImageNet runs should not construct an IS dataloader.")
    data_statistics["means"] = data_statistics["means"].to(device=device)

    train_iterator_state = None
    if not is_infinite:
        train_dataloader = train_temp
    else:
        raise NotImplementedError

    model = mutils.create_model(config).to(device=device)
    model.train()

    specified_log_const = _build_precond_specified_log_const(config, device)

    optimizer = losses_lib.get_optimizer(config, model.parameters())
    initial_step = 1

    if not is_infinite:
        model, optimizer, train_dataloader, val_dataloader, test_dataloader = (
            accelerator.prepare(
                model, optimizer, train_dataloader, val_dataloader, test_dataloader
            )
        )
        train_iter = None
    else:
        raise NotImplementedError

    scheduler = _create_lr_scheduler(config, optimizer, num_train_steps)

    use_ema = config.model.use_ema
    if use_ema:
        ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
        state = dict(optimizer=optimizer, scheduler=scheduler, model=model, ema=ema)
    else:
        state = dict(optimizer=optimizer, scheduler=scheduler, model=model)
    reset_lr_on_resume = bool(getattr(config.training, "reset_lr_on_resume", True))

    if config.model.fm_model_path == "none":
        fm_model = None
    else:
        fm_config = config.copy_and_resolve_references()
        if config.model.name == "precond_mnist_unet":
            fm_config.model.name = "mnist_unet"
        else:
            raise NotImplementedError
        fm_config.model.out_dim = "data"
        fm_model = mutils.create_model(fm_config).to(device=device)
        fm_model.load_state_dict(
            torch.load(config.model.fm_model_path, map_location=device)[
                "model_state_dict"
            ]
        )
        logging.info(f"Loaded flow matching model from {config.model.fm_model_path}")
        fm_model.eval()

    scaled_data_statistics = scaler_data_statistics(data_statistics)

    prob_path = get_prob_path(
        config=config,
        device=device,
        data_statistics=scaled_data_statistics,
        fm_model=fm_model,
    )

    main_build_noise_fn, reg_build_noise_fn, build_noise_fn, get_ebm, reg_estimator = (
        build_estimator_fns(
            config=config,
            prob_path=prob_path,
            dim=config.data.dim,
            data_shape=config.data.shape,
            specified_log_const=specified_log_const,
            force_fm_reg_estimator_none=True,
        )
    )

    train_step_fn = losses_lib.get_step_fn(
        estimator=estimator,
        num_noise_classes=num_noise_classes,
        num_cla_classes=config.estim.num_cla_classes,
        train=True,
        reg_estimator=reg_estimator,
        param=config.estim.param,
        use_ema=use_ema,
        main_weight=config.estim.main_weight,
        reg_weight=config.estim.reg_weight,
        combine_type=config.estim.combine_type,
        rne=config.estim.rne,
        grad_norm=config.training.grad_norm,
        accelerator=accelerator,
        prob_path=prob_path,
    )
    eval_loss_fn = losses_lib.get_loss_fn(
        estimator=estimator,
        reg_estimator=reg_estimator,
        param=config.estim.param,
        num_noise_classes=num_noise_classes,
        num_cla_classes=config.estim.num_cla_classes,
        main_weight=config.estim.main_weight,
        reg_weight=config.estim.reg_weight,
        combine_type=config.estim.combine_type,
        rne=config.estim.rne,
        prob_path=prob_path,
    )

    metrics = dict()
    train_time_between_eval_steps = dict()

    use_dequantization = config.data.use_dequantization
    selected_metrics = []
    best_metric_values = dict()
    selected_checkpoint_paths = dict()

    def reduce_train_log_values(metric_values, local_batch_size):
        weighted_values = torch.stack(
            [
                torch.as_tensor(value, device=device, dtype=torch.float64).detach()
                * local_batch_size
                for value in metric_values.values()
            ]
            + [torch.as_tensor(local_batch_size, device=device, dtype=torch.float64)]
        )
        weighted_values = accelerator.reduce(weighted_values, reduction="sum")
        total_samples = max(int(weighted_values[-1].item()), 1)
        return {
            metric_name: weighted_values[index].item() / total_samples
            for index, metric_name in enumerate(metric_values.keys())
        }

    assert estimator != "fm"
    assert data_dist is None

    selected_metrics = ["val_bpd", "val_loss"]
    selected_checkpoint_paths = {
        "val_bpd": os.path.join(checkpoints_dir, "best_val_bpd_model.pt"),
        "val_loss": os.path.join(checkpoints_dir, "best_val_loss_model.pt"),
    }
    best_metric_values = {metric_name: float("inf") for metric_name in selected_metrics}
    selected_metric = selected_metrics[0]

    if resume_checkpoint != "":
        resolved_resume_checkpoint = _resolve_resume_checkpoint_path(
            resume_checkpoint, accelerator.process_index, accelerator.num_processes
        )
        if not os.path.exists(resolved_resume_checkpoint):
            raise FileNotFoundError(
                f"Resume checkpoint not found at {resolved_resume_checkpoint}"
            )

        logging.info("Resuming training from checkpoint %s", resolved_resume_checkpoint)
        checkpoint = torch.load(
            resolved_resume_checkpoint, map_location=device, weights_only=False
        )

        checkpoint_num_processes = checkpoint.get("num_processes")
        if (checkpoint_num_processes is not None) and (
            checkpoint_num_processes != accelerator.num_processes
        ):
            logging.warning(
                "Checkpoint was created with %s process(es), but training is resuming "
                "with %s process(es). Exact dataloader replay may not be possible.",
                checkpoint_num_processes,
                accelerator.num_processes,
            )

        accelerator.unwrap_model(state["model"]).load_state_dict(
            checkpoint["model_state_dict"]
        )
        state["optimizer"].load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            state["scheduler"].load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            logging.warning(
                "Resume checkpoint %s does not contain scheduler state. "
                "Continuing with a freshly initialized scheduler.",
                resolved_resume_checkpoint,
            )
        if reset_lr_on_resume:
            old_lrs, new_lrs = _apply_config_learning_rate_to_resume(
                state["optimizer"], state["scheduler"], config.optim.lr
            )
            if accelerator.is_main_process:
                logging.info(
                    "Applied config.optim.lr=%s after resume; optimizer lr(s) %s -> %s",
                    float(config.optim.lr),
                    old_lrs,
                    new_lrs,
                )
        elif accelerator.is_main_process:
            logging.info(
                "Keeping optimizer lr(s) from checkpoint on resume: %s",
                [float(group["lr"]) for group in optimizer.param_groups],
            )

        if use_ema:
            if "ema_state_dict" not in checkpoint:
                raise ValueError(
                    "Resume checkpoint does not contain EMA parameters, but "
                    "config.model.use_ema is enabled."
                )
            state["ema"].load_state_dict(checkpoint["ema_state_dict"])

        grad_scaler = getattr(accelerator, "scaler", None)
        if (grad_scaler is not None) and ("grad_scaler_state_dict" in checkpoint):
            grad_scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
        elif (grad_scaler is not None) and ("grad_scaler_state_dict" not in checkpoint):
            logging.warning(
                "Resume checkpoint %s does not contain gradient scaler state.",
                resolved_resume_checkpoint,
            )

        initial_step = int(checkpoint["step"]) + 1
        metrics = checkpoint.get("metrics", metrics)
        best_metric_values = checkpoint.get("best_metric_values", best_metric_values)
        train_time_between_eval_steps = checkpoint.get(
            "train_time_between_eval_steps", train_time_between_eval_steps
        )
        start_time = _synchronized_perf_counter(device) - float(
            checkpoint.get("accumulated_train_time", 0.0)
        )

        checkpoint_rng_state = checkpoint.get("rng_state")
        if not is_infinite:
            restored_train_iterator_state = checkpoint.get("train_iterator_state")
            process_count_mismatch = (checkpoint_num_processes is not None) and (
                checkpoint_num_processes != accelerator.num_processes
            )
            can_restore_iterator = (restored_train_iterator_state is not None) and (
                not process_count_mismatch
            )
            if can_restore_iterator:
                train_iterator_state = restored_train_iterator_state
                train_iter = _restore_train_iterator(
                    train_dataloader, train_iterator_state
                )
            elif process_count_mismatch:
                logging.warning(
                    "Skipping exact dataloader replay for resume from %s because "
                    "the checkpoint was saved with %s process(es), but this run has "
                    "%s process(es). Restarting from a fresh train iterator.",
                    resolved_resume_checkpoint,
                    checkpoint_num_processes,
                    accelerator.num_processes,
                )
                if checkpoint_rng_state is not None:
                    _set_rng_state(checkpoint_rng_state)
                train_iter, train_iterator_state = _create_train_iterator(
                    train_dataloader
                )
            elif restored_train_iterator_state is None:
                logging.warning(
                    "Resume checkpoint %s does not contain training iterator state. "
                    "Falling back to a fresh iterator.",
                    resolved_resume_checkpoint,
                )
                if checkpoint_rng_state is not None:
                    _set_rng_state(checkpoint_rng_state)
                train_iter, train_iterator_state = _create_train_iterator(
                    train_dataloader
                )

        if checkpoint_rng_state is None:
            logging.warning(
                "Resume checkpoint %s does not contain RNG state. Subsequent "
                "training may diverge from the original run.",
                resolved_resume_checkpoint,
            )
        else:
            _set_rng_state(checkpoint_rng_state)
        accelerator.wait_for_everyone()

    def evaluate_fn(step, model, metrics):
        eval_model = (
            accelerator.unwrap_model(model) if (accelerator is not None) else model
        )
        return image_evaluate(
            set_names=["val"],
            set_dataloaders=[val_dataloader],
            is_dataloader=None,
            device=device,
            use_dequantization=use_dequantization,
            build_noise_fn=build_noise_fn,
            eval_loss_fn=eval_loss_fn,
            data_dist=data_dist,
            get_ebm=get_ebm,
            step=step,
            model=eval_model,
            metrics=metrics,
            accelerator=accelerator,
            non_blocking=config.eval.non_blocking,
        )

    if reg_estimator in ["dsm", "fm"]:

        def train_log_fn(step, losses, local_batch_size):
            main_loss, reg_loss, loss = losses
            if use_wandb and (step % train_log_freq == 0):
                averaged_metrics = reduce_train_log_values(
                    {
                        "train_main_loss": main_loss,
                        "train_reg_loss": reg_loss,
                        "train_loss": loss,
                    },
                    local_batch_size,
                )
            if (
                use_wandb
                and accelerator.is_main_process
                and (step % train_log_freq == 0)
            ):
                averaged_metrics["lr"] = optimizer.param_groups[0]["lr"]
                wandb.log(
                    averaged_metrics,
                    step=step,
                )

    else:

        def train_log_fn(step, loss, local_batch_size):
            if use_wandb and (step % train_log_freq == 0):
                averaged_metrics = reduce_train_log_values(
                    {
                        "train_main_loss": loss,
                        "train_reg_loss": torch.zeros_like(loss),
                        "train_loss": loss,
                    },
                    local_batch_size,
                )
            if (
                use_wandb
                and accelerator.is_main_process
                and (step % train_log_freq == 0)
            ):
                averaged_metrics["lr"] = optimizer.param_groups[0]["lr"]
                wandb.log(
                    averaged_metrics,
                    step=step,
                )

    def eval_log_fn(step, metrics):
        all_current_metrics = {
            metric_name: metrics[metric_name][step] for metric_name in metrics.keys()
        }
        logged_metrics = {
            metric_name: metric_value
            for metric_name, metric_value in all_current_metrics.items()
            if not metric_name.startswith("test")
        }
        logging.info(
            "step: %s, %s",
            step,
            ", ".join(
                f"{metric_name}: {metric_value:.4f}"
                for metric_name, metric_value in logged_metrics.items()
            ),
        )
        if use_wandb:
            wandb.log(
                logged_metrics,
                step=step,
            )

    train_log_freq = config.training.log_freq
    eval_freq = config.eval.eval_freq
    if (accelerator.num_processes > 1) and (resume_checkpoint == ""):
        seed_rank_rng(config.seed, accelerator.process_index)

    if train_iter is None:
        train_iter, train_iterator_state = _create_train_iterator(train_dataloader)

    if accelerator.is_main_process:
        last_eval_end_time = _synchronized_perf_counter(device)

    non_blocking = bool(getattr(config.training, "non_blocking", False))
    previous_save_all_checkpoint_step = None

    for step in range(initial_step, num_train_steps + 1):
        # non_blocking is added for imagenet
        try:
            data = next(train_iter)[0].to(device=device, non_blocking=non_blocking)
        except StopIteration:
            # infinite dataloader will not come here
            train_iter, train_iterator_state = _create_train_iterator(train_dataloader)
            data = next(train_iter)[0].to(device=device, non_blocking=non_blocking)

        if train_iterator_state is not None:
            train_iterator_state["steps_into_epoch"] += 1

        # dequantization and scale
        if use_dequantization:
            data = ((data * 255.0) + torch.rand_like(data)) / 256.0
        data = scaler(data)

        losses = train_step_fn(state, data, build_noise_fn)
        scheduler.step()

        train_log_fn(step, losses, data.shape[0])

        if step % eval_freq == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                train_time_between_eval_steps[step] = (
                    _synchronized_perf_counter(device) - last_eval_end_time
                )
            with torch.no_grad():
                model = state["model"]
                was_training = model.training
                if use_ema:
                    ema = state["ema"]
                    ema.store(model.parameters())
                    ema.copy_to(model.parameters())
                model.eval()

                metrics = evaluate_fn(step, model, metrics)

                if use_ema:
                    ema.restore(model.parameters())
                if was_training:
                    model.train()

            improved_metrics = []
            for metric_name in selected_metrics:
                current_metric = metrics[metric_name][step]
                if current_metric < best_metric_values[metric_name]:
                    best_metric_values[metric_name] = current_metric
                    improved_metrics.append(metric_name)

            accelerator.wait_for_everyone()
            if save_checkpoint or save_all_checkpoints:
                latest_checkpoint_state = _build_training_checkpoint(
                    accelerator=accelerator,
                    state=state,
                    step=step,
                    metrics=metrics,
                    best_metric_values=best_metric_values,
                    train_time_between_eval_steps=train_time_between_eval_steps,
                    train_iterator_state=train_iterator_state,
                    start_time=start_time,
                    device=device,
                )
                torch.save(latest_checkpoint_state, latest_process_checkpoint_path)
                if accelerator.is_main_process and (
                    latest_process_checkpoint_path != latest_checkpoint_path
                ):
                    torch.save(latest_checkpoint_state, latest_checkpoint_path)

            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                if save_checkpoint:
                    logging.info(
                        "Saved latest resume checkpoint to %s",
                        latest_checkpoint_path,
                    )
                eval_log_fn(step, metrics)

                for metric_name in improved_metrics:
                    current_metric = metrics[metric_name][step]
                    logging.info(
                        f"New best {metric_name} found at step {step} with {current_metric:.4f}"
                    )

            if save_checkpoint:
                for metric_name in improved_metrics:
                    best_checkpoint_path = selected_checkpoint_paths[metric_name]
                    best_process_checkpoint_path = _get_process_checkpoint_path(
                        best_checkpoint_path,
                        accelerator.process_index,
                        accelerator.num_processes,
                    )
                    torch.save(latest_checkpoint_state, best_process_checkpoint_path)
                    if accelerator.is_main_process and (
                        best_process_checkpoint_path != best_checkpoint_path
                    ):
                        torch.save(latest_checkpoint_state, best_checkpoint_path)

            if save_all_checkpoints:
                step_checkpoint_path = os.path.join(
                    checkpoints_dir, f"step_{step}_model.pt"
                )
                step_process_checkpoint_path = _get_process_checkpoint_path(
                    step_checkpoint_path,
                    accelerator.process_index,
                    accelerator.num_processes,
                )
                torch.save(latest_checkpoint_state, step_process_checkpoint_path)
                if accelerator.is_main_process and (
                    step_process_checkpoint_path != step_checkpoint_path
                ):
                    torch.save(
                        latest_checkpoint_state,
                        step_checkpoint_path,
                    )

            accelerator.wait_for_everyone()
            if (
                save_all_checkpoints
                and accelerator.is_main_process
                and previous_save_all_checkpoint_step is not None
            ):
                previous_checkpoint_path = os.path.join(
                    checkpoints_dir,
                    f"step_{previous_save_all_checkpoint_step}_model.pt",
                )
                for stale_checkpoint_path in _get_non_main_process_checkpoint_paths(
                    previous_checkpoint_path, accelerator.num_processes
                ):
                    try:
                        os.remove(stale_checkpoint_path)
                    except FileNotFoundError:
                        pass
                    except OSError:
                        logging.warning(
                            "Failed to remove stale checkpoint %s",
                            stale_checkpoint_path,
                            exc_info=True,
                        )
            if save_all_checkpoints:
                previous_save_all_checkpoint_step = step
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                last_eval_end_time = _synchronized_perf_counter(device)

    metrics["best"] = dict()
    selected_metric_values = metrics[selected_metric]
    best_step = min(selected_metric_values, key=selected_metric_values.get)
    metrics["best"]["step"] = best_step
    for metric_name in selected_metrics:
        if metric_name != selected_metric:
            metric_values = metrics[metric_name]
            metric_best_step = min(metric_values, key=metric_values.get)
            logging.info(
                f"Best {metric_name} at step {metric_best_step}: {metric_values[metric_best_step]:.6f}"
            )
            metrics["best"][f"{metric_name}_step"] = metric_best_step
    for metric_name, metric_values in metrics.items():
        if metric_name.startswith("val"):
            raw_metric_name = metric_name.replace("val_", "")
            logging.info(
                f"{raw_metric_name} val set result: {metric_values[best_step]:.3f} at {best_step}"
            )
            metrics["best"][metric_name] = metric_values[best_step]
            if config.print_test and f"test_{raw_metric_name}" in metrics:
                test_metric_values = metrics[f"test_{raw_metric_name}"]
                logging.info(
                    f"{raw_metric_name} test set result: {test_metric_values[best_step]:.3f} at {best_step}"
                )
                metrics["best"][f"test_{raw_metric_name}"] = test_metric_values[
                    best_step
                ]

    accelerator.wait_for_everyone()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    if accelerator.is_main_process:
        end_time = _synchronized_perf_counter(device)
        times = [start_time, end_time, end_time - start_time]
        metrics["times"] = times
        metrics["train_time_between_eval_steps"] = train_time_between_eval_steps
        with open(f"{workdir}/metrics.pkl", "wb") as f:
            pickle.dump(metrics, f)


def evaluate(config, workdir, eval_folder):
    seed_all(config.seed, exact=(config.data.name != "imagenet64"))

    evaluate_bpd = bool(config.eval.bpd)
    evaluate_samples = bool(config.eval.sample)
    if not evaluate_bpd and not evaluate_samples:
        logging.warning(
            "Skipping evaluation because both config.eval.bpd and "
            "config.eval.sample are false."
        )
        return

    mixed_precision = getattr(config.training, "mixed_precision", "no")
    if mixed_precision not in {"no", "fp16", "bf16"}:
        raise ValueError(
            "config.training.mixed_precision must be one of 'no', 'fp16', or "
            f"'bf16', got {mixed_precision!r}"
        )
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        dataloader_config=DataLoaderConfiguration(
            split_batches=config.training.split_batches,
            even_batches=config.training.even_batches,
            use_seedable_sampler=True,
        ),
    )
    device = accelerator.device
    if accelerator.is_main_process:
        logging.info(
            "Accelerate eval runtime: mixed_precision=%s, device=%s, num_processes=%s",
            accelerator.mixed_precision,
            accelerator.device,
            accelerator.num_processes,
        )

    checkpoint_path = _resolve_eval_checkpoint_path(
        workdir, config.eval.checkpoint_name
    )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Evaluation checkpoint not found at {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = mutils.create_model(config).to(device=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    use_ema = config.model.use_ema
    if use_ema:
        if "ema_state_dict" not in checkpoint.keys():
            raise ValueError(
                "Evaluation is configured to use EMA parameters, but checkpoint "
                f"{checkpoint_path} does not contain `ema_state_dict`."
            )
        ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
        ema.load_state_dict(checkpoint["ema_state_dict"])
        ema.copy_to(model.parameters())
        logging.info("Loaded EMA parameters from %s for evaluation.", checkpoint_path)

    model.eval()

    specified_log_const = _build_precond_specified_log_const(config, device)

    data_dist = None
    test_dataloader = None
    if evaluate_bpd:
        (
            _,
            _,
            test_dataloader,
            is_dataloader,
            _,
            data_dist,
            data_statistics,
        ) = create_datasets(
            config,
            device=device,
            process_index=accelerator.process_index,
            train=False,
        )
        if is_dataloader is not None:
            raise RuntimeError("ImageNet runs should not construct an IS dataloader.")
        if data_dist is not None:
            raise RuntimeError("ImageNet BPD evaluation expects no analytic data_dist.")
    else:
        data_statistics = _build_imagenet_data_statistics(config, device)

    if "means" in data_statistics and torch.is_tensor(data_statistics["means"]):
        data_statistics["means"] = data_statistics["means"].to(device=device)

    scaled_data_statistics = scaler_data_statistics(data_statistics)

    prob_path = get_prob_path(
        config=config,
        device=device,
        data_statistics=scaled_data_statistics,
        fm_model=None,
    )

    eval_dir = os.path.join(workdir, eval_folder)
    if accelerator.is_main_process:
        os.makedirs(eval_dir, exist_ok=True)

    if evaluate_bpd:
        (
            main_build_noise_fn,
            reg_build_noise_fn,
            build_noise_fn,
            get_ebm,
            reg_estimator,
        ) = build_estimator_fns(
            config=config,
            prob_path=prob_path,
            dim=config.data.dim,
            data_shape=config.data.shape,
            specified_log_const=specified_log_const,
            force_fm_reg_estimator_none=True,
        )

        eval_loss_fn = losses_lib.get_loss_fn(
            estimator=config.estim.name,
            reg_estimator=reg_estimator,
            param=config.estim.param,
            num_noise_classes=config.estim.num_noise_classes,
            num_cla_classes=config.estim.num_cla_classes,
            main_weight=config.estim.main_weight,
            reg_weight=config.estim.reg_weight,
            combine_type=config.estim.combine_type,
            rne=config.estim.rne,
            prob_path=prob_path,
        )

        test_eval_specs = [
            {
                "set_name": "test_with_raw",
                "output_dir": "test_with_raw",
                "horizontal_flip": False,
                "horizontal_flip_probability": 0.0,
            },
            {
                "set_name": "test_with_random_horizontal_flip",
                "output_dir": "test_with_random_horizontal_flip",
                "horizontal_flip": True,
                "horizontal_flip_probability": 0.5,
            },
            {
                "set_name": "test_with_always_horizontal_flip",
                "output_dir": "test_with_always_horizontal_flip",
                "horizontal_flip": True,
                "horizontal_flip_probability": 1.0,
            },
        ]
        test_dataloaders = [
            _add_sample_indices_to_dataloader(test_dataloader[spec["set_name"]])
            for spec in test_eval_specs
        ]
        prepared = accelerator.prepare(model, *test_dataloaders)
        model = prepared[0]
        test_dataloaders = list(prepared[1:])
        eval_model = accelerator.unwrap_model(model)
        eval_model.eval()
        step = int(checkpoint.get("step", 0))
        metrics = {}

        metrics, log_density_outputs = image_evaluate(
            set_names=[spec["set_name"] for spec in test_eval_specs],
            set_dataloaders=test_dataloaders,
            is_dataloader=None,
            device=device,
            use_dequantization=config.data.use_dequantization,
            build_noise_fn=build_noise_fn,
            eval_loss_fn=eval_loss_fn,
            data_dist=data_dist,
            get_ebm=get_ebm,
            step=step,
            model=eval_model,
            metrics=metrics,
            accelerator=accelerator,
            non_blocking=config.eval.non_blocking,
            return_log_densities=True,
        )

        summary_metrics = {
            "checkpoint_path": checkpoint_path,
            "metrics": metrics,
            "step": step,
            "results": {},
        }

        for spec in test_eval_specs:
            set_name = spec["set_name"]
            test_log_density_output = log_density_outputs[set_name]
            test_log_density_indices = test_log_density_output["indices"]
            test_log_densities = test_log_density_output["log_densities"]

            output_dir = os.path.join(eval_dir, spec["output_dir"])
            summary_path = os.path.join(output_dir, "test_bpd.pkl")
            log_density_path = os.path.join(output_dir, "test_log_densities.pt")
            bpd_metric_name = f"{set_name}_bpd"
            variant_summary_metrics = {
                "checkpoint_path": checkpoint_path,
                "metrics": metrics,
                "step": step,
                "set_name": set_name,
                "random_horizontal_flip": spec["horizontal_flip"],
                "horizontal_flip_probability": spec.get("horizontal_flip_probability"),
                "test_log_density_indices": test_log_density_indices,
                "test_log_densities": test_log_densities,
                "test_log_density_path": log_density_path,
            }
            if bpd_metric_name in metrics:
                variant_summary_metrics["test_bpd"] = metrics[bpd_metric_name][step]
                variant_summary_metrics[bpd_metric_name] = metrics[bpd_metric_name][
                    step
                ]

            summary_metrics["results"][set_name] = variant_summary_metrics

        if accelerator.is_main_process:
            current_metrics = {
                metric_name: metric_values[step]
                for metric_name, metric_values in metrics.items()
                if isinstance(metric_values, dict) and step in metric_values
            }
            logging.info(
                "ImageNet test metrics at step %s: %s",
                step,
                ", ".join(
                    f"{metric_name}: {metric_value:.6f}"
                    for metric_name, metric_value in current_metrics.items()
                ),
            )
            for spec in test_eval_specs:
                set_name = spec["set_name"]
                variant_summary_metrics = summary_metrics["results"][set_name]
                output_dir = os.path.join(eval_dir, spec["output_dir"])
                summary_path = os.path.join(output_dir, "test_bpd.pkl")
                log_density_path = os.path.join(output_dir, "test_log_densities.pt")
                os.makedirs(output_dir, exist_ok=True)
                with open(summary_path, "wb") as f:
                    pickle.dump(variant_summary_metrics, f)
                torch.save(
                    {
                        "indices": variant_summary_metrics["test_log_density_indices"],
                        "log_densities": variant_summary_metrics["test_log_densities"],
                        "set_name": set_name,
                        "random_horizontal_flip": spec["horizontal_flip"],
                        "horizontal_flip_probability": spec.get(
                            "horizontal_flip_probability"
                        ),
                    },
                    log_density_path,
                )
                logging.info(
                    "Saved %s ImageNet %s log densities to %s",
                    variant_summary_metrics["test_log_densities"].numel(),
                    set_name,
                    log_density_path,
                )
                logging.info(
                    "Saved ImageNet %s evaluation summary to %s",
                    set_name,
                    summary_path,
                )
        accelerator.wait_for_everyone()

    if evaluate_samples:
        total_samples = int(getattr(config.eval, "sample_num_samples", 64))
        sample_batch_size = int(getattr(config.eval, "sample_batch_size", 8))
        sample_num_steps = int(getattr(config.eval, "sample_num_steps", 1000))
        if total_samples <= 0:
            raise ValueError(
                f"config.eval.sample_num_samples must be positive, got {total_samples}."
            )
        if sample_batch_size <= 0:
            raise ValueError(
                "config.eval.sample_batch_size must be positive, got "
                f"{sample_batch_size}."
            )
        if sample_num_steps <= 0:
            raise ValueError(
                "config.eval.sample_num_steps must be positive, got "
                f"{sample_num_steps}."
            )

        if accelerator.num_processes > 1:
            seed_rank_rng(config.seed, accelerator.process_index)

        local_num_samples = _get_local_count(
            total_samples, accelerator.process_index, accelerator.num_processes
        )
        velocity_wrapper = VelocityWrapper(
            model=model,
            prob_path=prob_path,
        )

        accelerator.wait_for_everyone()
        sample_start_time = _synchronized_perf_counter(device)
        if local_num_samples > 0:
            local_samples = euler_sample(
                velocity_wrapper=velocity_wrapper,
                num_samples=local_num_samples,
                batch_size=min(sample_batch_size, local_num_samples),
                data_shape=config.data.shape,
                num_steps=sample_num_steps,
                device=device,
            )
        else:
            local_samples = torch.empty((0, *config.data.shape), device=device)
        local_sample_time = _synchronized_perf_counter(device) - sample_start_time
        gathered_sample_times = accelerator.gather(
            torch.tensor([local_sample_time], device=device, dtype=torch.float64)
        )
        sample_time = float(gathered_sample_times.max().item())

        samples = _gather_variable_batch_to_main(
            local_batch=local_samples,
            local_count=local_num_samples,
            total_count=total_samples,
            accelerator=accelerator,
        )
        if accelerator.is_main_process:
            samples = inverse_scaler(samples)
            sample_path = os.path.join(eval_dir, "samples.png")
            save_image(samples[:64], sample_path)
            samples_per_second = (
                total_samples / sample_time if sample_time > 0.0 else float("inf")
            )
            sampling_metrics = {
                "checkpoint_path": checkpoint_path,
                "step": int(checkpoint.get("step", 0)),
                "num_samples": total_samples,
                "batch_size_per_process": sample_batch_size,
                "num_sampling_steps": sample_num_steps,
                "num_processes": accelerator.num_processes,
                "sample_generation_time_seconds": sample_time,
                "samples_per_second": samples_per_second,
                "per_process_generation_time_seconds": (
                    gathered_sample_times.detach().cpu().tolist()
                ),
                "sample_path": sample_path,
            }
            sampling_metrics_path = os.path.join(eval_dir, "sampling_metrics.pkl")
            with open(sampling_metrics_path, "wb") as f:
                pickle.dump(sampling_metrics, f)
            logging.info(
                "Generated %s ImageNet samples in %.3f seconds (%.3f samples/second).",
                samples.shape[0],
                sample_time,
                samples_per_second,
            )
            logging.info(
                "Saved ImageNet samples to %s and sampling metrics to %s",
                sample_path,
                sampling_metrics_path,
            )

        accelerator.wait_for_everyone()
