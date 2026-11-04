import os
import time
import shutil
import math
import torch
import logging
import losses_lib
import pickle
from accelerate import Accelerator, DataLoaderConfiguration
from torch.optim import lr_scheduler

from models.ema import ExponentialMovingAverage
import wandb

# for registrations
from models import unet

from models import model_utils as mutils
from data.data import (
    create_datasets,
    prepare_data_cache,
    prepare_data_sets,
)
from absl import flags
from estimator_utils import build_estimator_fns
from utils import (
    seed_all,
    seed_rank_rng,
    scaler,
    inverse_scaler,
    scaler_data_statistics,
    image_evaluate,
    image_evaluate_fm,
    VelocityWrapper,
    euler_sample,
)

from torchvision.utils import save_image
from prob_paths_lib import get_prob_path
from analytics import gaussian_mixture_score

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


def train(config, workdir):
    start_time = _synchronized_perf_counter(torch.device(config.device))
    seed_all(config.seed, exact=(config.data.name == "mnist_mixture"))

    num_train_steps = config.training.num_iters

    # do not use accelerator for setup and sets
    if config.training.setup:
        prepare_data_cache(config)
        return

    if num_train_steps == 0:
        prepare_data_sets(config)
        return

    accelerator = Accelerator(
        split_batches=config.training.split_batches,
        dataloader_config=DataLoaderConfiguration(
            even_batches=config.training.even_batches
        ),
    )
    device = accelerator.device
    estimator = config.estim.name
    reg_estimator = config.estim.reg_estimator

    num_noise_classes = config.estim.num_noise_classes
    use_wandb = config.wandb.use_wandb
    save_checkpoint = config.training.save_checkpoint
    save_all_checkpoints = config.training.save_all_checkpoints

    if save_all_checkpoints:
        all_checkpoints_dir = os.path.join(workdir, "checkpoints")
        if accelerator.is_main_process:
            if os.path.exists(all_checkpoints_dir):
                shutil.rmtree(all_checkpoints_dir)
            os.mkdir(all_checkpoints_dir)
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
    data_statistics["means"] = data_statistics["means"].to(device=device)

    if not is_infinite:
        train_dataloader = train_temp
        train_iter = iter(train_dataloader)
    else:
        train_iter = train_temp

    model = mutils.create_model(config).to(device=device)
    model.train()

    specified_log_const = _build_precond_specified_log_const(config, device)

    optimizer = losses_lib.get_optimizer(config, model.parameters())
    initial_step = 1

    if not is_infinite:
        if is_dataloader is not None:
            (
                model,
                optimizer,
                train_dataloader,
                val_dataloader,
                test_dataloader,
                is_dataloader,
            ) = accelerator.prepare(
                model,
                optimizer,
                train_dataloader,
                val_dataloader,
                test_dataloader,
                is_dataloader,
            )
        else:
            model, optimizer, train_dataloader, val_dataloader, test_dataloader = (
                accelerator.prepare(
                    model, optimizer, train_dataloader, val_dataloader, test_dataloader
                )
            )
        train_iter = iter(train_dataloader)
    else:
        if is_dataloader is not None:
            model, optimizer, val_dataloader, test_dataloader, is_dataloader = (
                accelerator.prepare(
                    model, optimizer, val_dataloader, test_dataloader, is_dataloader
                )
            )
        else:
            model, optimizer, val_dataloader, test_dataloader = accelerator.prepare(
                model, optimizer, val_dataloader, test_dataloader
            )

    scheduler = _create_lr_scheduler(config, optimizer, num_train_steps)

    use_ema = config.model.use_ema
    if use_ema:
        ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
        state = dict(optimizer=optimizer, scheduler=scheduler, model=model, ema=ema)
    else:
        state = dict(optimizer=optimizer, scheduler=scheduler, model=model)

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
        mixture_loss=getattr(config.estim, "mixture_loss", False),
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
        mixture_loss=getattr(config.estim, "mixture_loss", False),
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

    if estimator != "fm":
        if data_dist is not None:
            selected_metrics = ["val_mse", "val_norm_mse", "val_loss"]
            selected_checkpoint_paths = {
                "val_mse": os.path.join(workdir, "best_val_mse_model.pt"),
                "val_norm_mse": os.path.join(workdir, "best_val_norm_mse_model.pt"),
                "val_loss": os.path.join(workdir, "best_val_loss_model.pt"),
            }
        else:
            selected_metrics = ["val_bpd", "val_loss"]
            selected_checkpoint_paths = {
                "val_bpd": os.path.join(workdir, "best_val_bpd_model.pt"),
                "val_loss": os.path.join(workdir, "best_val_loss_model.pt"),
            }
        best_metric_values = {
            metric_name: float("inf") for metric_name in selected_metrics
        }
        selected_metric = selected_metrics[0]

        def evaluate_fn(step, model, metrics):
            # added to make mnist work
            eval_model = (
                accelerator.unwrap_model(model) if (accelerator is not None) else model
            )
            return image_evaluate(
                set_names=["val", "test"],
                set_dataloaders=[val_dataloader, test_dataloader],
                is_dataloader=is_dataloader,
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
                    averaged_metrics["lr"] = scheduler.get_last_lr()[0]
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
                    averaged_metrics["lr"] = scheduler.get_last_lr()[0]
                    wandb.log(
                        averaged_metrics,
                        step=step,
                    )

        def eval_log_fn(step, metrics):
            all_current_metrics = {
                metric_name: metrics[metric_name][step]
                for metric_name in metrics.keys()
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

    else:
        selected_metric = "val_loss"
        selected_metrics = [selected_metric]
        best_metric_values = {selected_metric: float("inf")}
        selected_checkpoint_paths = {
            selected_metric: os.path.join(workdir, "best_model.pt")
        }

        gt_score_fn = None
        if config.data.name == "mnist_mixture":
            data_means = scaled_data_statistics["means"]
            flat_data_means = data_means.view((data_means.shape[0], -1))
            data_std = scaled_data_statistics["std"]

            def gt_score_fn(x, t):
                return gaussian_mixture_score(
                    x=x.view((x.shape[0], -1)),
                    t=t,
                    mus=flat_data_means,
                    sigma=data_std,
                    alpha_fn=prob_path.alpha,
                    beta_fn=prob_path.beta,
                ).view((x.shape[0], *config.data.shape))

        def evaluate_fn(step, model, metrics):
            return image_evaluate_fm(
                prob_path=prob_path,
                gt_score_fn=gt_score_fn,
                set_names=["val", "test"],
                set_dataloaders=[val_dataloader, test_dataloader],
                step=step,
                model=model,
                build_noise_fn=build_noise_fn,
                eval_loss_fn=eval_loss_fn,
                metrics=metrics,
                device=device,
                use_dequantization=use_dequantization,
                accelerator=accelerator,
            )

        def train_log_fn(step, losses, local_batch_size):
            loss = losses
            if use_wandb and (step % train_log_freq == 0):
                averaged_metrics = reduce_train_log_values(
                    {"train_loss": loss},
                    local_batch_size,
                )
            if (
                use_wandb
                and accelerator.is_main_process
                and (step % train_log_freq == 0)
            ):
                averaged_metrics["lr"] = scheduler.get_last_lr()[0]
                wandb.log(
                    averaged_metrics,
                    step=step,
                )

        def eval_log_fn(step, metrics):
            all_current_metrics = {
                metric_name: metrics[metric_name][step]
                for metric_name in metrics.keys()
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
    if accelerator.num_processes > 1:
        seed_rank_rng(config.seed, accelerator.process_index)

    if accelerator.is_main_process:
        last_eval_end_time = _synchronized_perf_counter(device)

    for step in range(initial_step, num_train_steps + 1):
        try:
            data = next(train_iter)[0].to(device=device)
        except StopIteration:
            # infinite dataloader will not come here
            train_iter = iter(train_dataloader)
            data = next(train_iter)[0].to(device=device)

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

            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                eval_log_fn(step, metrics)

                for metric_name in selected_metrics:
                    current_metric = metrics[metric_name][step]
                    if current_metric < best_metric_values[metric_name]:
                        best_metric_values[metric_name] = current_metric
                        logging.info(
                            f"New best {metric_name} found at step {step} with {current_metric:.4f}"
                        )
                        if save_checkpoint:
                            unwrapped_model = accelerator.unwrap_model(model)
                            checkpoint_state = {
                                "step": step,
                                "model_state_dict": unwrapped_model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "scheduler_state_dict": scheduler.state_dict(),
                                # **all_current_metrics,
                            }
                            if use_ema:
                                checkpoint_state["ema_state_dict"] = ema.state_dict()
                            torch.save(
                                checkpoint_state, selected_checkpoint_paths[metric_name]
                            )

                if save_all_checkpoints:
                    unwrapped_model = accelerator.unwrap_model(model)
                    checkpoint_state = {
                        "step": step,
                        "model_state_dict": unwrapped_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        # **all_current_metrics,
                    }
                    if use_ema:
                        checkpoint_state["ema_state_dict"] = ema.state_dict()
                    torch.save(
                        checkpoint_state,
                        os.path.join(all_checkpoints_dir, f"step_{step}_model.pt"),
                    )
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


def _get_eval_time_grid(config, device):
    default_time_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    raw_time_grid = (
        config.eval.time_grid if "time_grid" in config.eval else default_time_grid
    )
    time_grid = torch.as_tensor(raw_time_grid, device=device, dtype=torch.float32)

    if time_grid.ndim != 1:
        raise ValueError(
            f"config.eval.time_grid must be 1D, got shape {tuple(time_grid.shape)}."
        )
    if time_grid.numel() == 0:
        raise ValueError("config.eval.time_grid must contain at least one time value.")
    if torch.any((time_grid < 0.0) | (time_grid > 1.0)):
        raise ValueError(
            "config.eval.time_grid must lie in [0, 1], "
            f"got {time_grid.detach().cpu().tolist()}."
        )

    return time_grid


def evaluate(config, workdir, eval_folder):
    seed_all(config.seed)

    device = config.device

    checkpoint_path = f"{workdir}/{config.eval.checkpoint_name}"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    with open(f"{workdir}/metrics.pkl", "rb") as f:
        metrics = pickle.load(f)

    if config.eval.checkpoint_name == "best_val_norm_mse_model.pt":
        step = metrics["best"]["val_norm_mse_step"]
        log_constant = metrics["log_constant"][step]
        print(log_constant)
    else:
        log_constant = 0.0

    model = mutils.create_model(config).to(device=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    use_ema = config.model.use_ema
    if use_ema:
        if "ema_state_dict" not in checkpoint.keys():
            logging.warning(
                "Checkpoint %s does not contain EMA parameters. Falling back to "
                "raw model parameters for evaluation.",
                checkpoint_path,
            )
        else:
            ema = ExponentialMovingAverage(
                model.parameters(), decay=config.model.ema_rate
            )
            ema.load_state_dict(checkpoint["ema_state_dict"])
            ema.copy_to(model.parameters())
            logging.info(
                "Loaded EMA parameters from %s for evaluation.", checkpoint_path
            )

    model.eval()

    specified_log_const = _build_precond_specified_log_const(config, device)

    (
        train_temp,
        val_dataloader,
        test_dataloader,
        is_dataloader,
        is_infinite,
        data_dist,
        data_statistics,
    ) = create_datasets(config, device=device)

    scaled_data_statistics = scaler_data_statistics(data_statistics)

    prob_path = get_prob_path(
        config=config,
        device=device,
        data_statistics=scaled_data_statistics,
        fm_model=None,
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

    ebm_model = get_ebm(model)

    eval_dir = os.path.join(workdir, eval_folder)
    os.makedirs(eval_dir, exist_ok=True)

    if config.eval.ddplots:
        from utils import scaler_log_det
        from analytics import time_varying_gmm_log_prob
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        time_grid = _get_eval_time_grid(config, device=device)

        use_dequantization = config.data.use_dequantization
        if ("means" not in scaled_data_statistics) or (
            "std" not in scaled_data_statistics
        ):
            raise ValueError(
                "Time-varying log-density evaluation requires GMM data statistics "
                "with `means` and `std`."
            )
        flat_scaled_means = scaled_data_statistics["means"].view(
            scaled_data_statistics["means"].shape[0], -1
        )
        num_components = flat_scaled_means.shape[0]
        gmm_sigma = scaled_data_statistics["std"]

        per_time_results = []

        with torch.no_grad():
            for t_value in time_grid:
                gt_log_probs = []
                model_log_probs = []
                scalar_t = float(t_value.item())

                for batch in test_dataloader:
                    batch_data = batch[0].to(device=device)

                    if use_dequantization:
                        batch_data = (
                            (batch_data * 255.0) + torch.rand_like(batch_data)
                        ) / 256.0

                    scaled_batch_data = scaler(batch_data)
                    batch_ts = torch.full(
                        (batch_data.shape[0],),
                        fill_value=scalar_t,
                        device=device,
                        dtype=scaled_batch_data.dtype,
                    )
                    perturbed_batch_data = prob_path.sample(
                        z=scaled_batch_data, t=batch_ts
                    )
                    flat_perturbed_batch_data = perturbed_batch_data.view(
                        perturbed_batch_data.shape[0], -1
                    )

                    batch_gt_log_probs = time_varying_gmm_log_prob(
                        y=flat_perturbed_batch_data,
                        means=flat_scaled_means,
                        sigma=gmm_sigma,
                        alpha_t=prob_path.alpha(batch_ts),
                        beta_t=prob_path.beta(batch_ts),
                        num_components=num_components,
                    ).reshape(-1) + config.data.dim * math.log(2.0)

                    if scalar_t == 0.0:
                        ebm_t = config.estim.eps1
                    elif scalar_t == 1.0:
                        ebm_t = 1.0 - config.estim.eps2
                    else:
                        ebm_t = scalar_t

                    batch_model_log_probs = (
                        -ebm_model(perturbed_batch_data, t=ebm_t).squeeze(dim=-1)
                        + scaler_log_det(batch_data)
                        - log_constant
                    )

                    gt_log_probs.append(batch_gt_log_probs.detach().cpu())
                    model_log_probs.append(batch_model_log_probs.detach().cpu())

                gt_log_probs = torch.cat(gt_log_probs)
                model_log_probs = torch.cat(model_log_probs)
                correlation = torch.corrcoef(
                    torch.stack(
                        [
                            gt_log_probs.to(torch.float64),
                            model_log_probs.to(torch.float64),
                        ],
                        dim=0,
                    )
                )[0, 1].item()

                per_time_results.append(
                    {
                        "time": scalar_t,
                        "correlation": correlation,
                        "gt_log_probs": gt_log_probs,
                        "model_log_probs": model_log_probs,
                    }
                )

        summary_metrics = {
            "time_grid": [result["time"] for result in per_time_results],
            "correlations": [result["correlation"] for result in per_time_results],
        }

        summary_path = os.path.join(eval_dir, "test_log_density_time_grid.pkl")
        with open(summary_path, "wb") as f:
            pickle.dump(summary_metrics, f)

        num_times = len(per_time_results)
        panel_size = 4.8
        fig, axes = plt.subplots(
            1,
            num_times,
            figsize=(panel_size * num_times, panel_size + 0.9),
            squeeze=False,
            constrained_layout=True,
        )
        fig.set_constrained_layout_pads(
            w_pad=0.03,
            h_pad=0.03,
            wspace=0.05,
            hspace=0.02,
        )

        label_fontsize = 18
        title_fontsize = 18
        tick_fontsize = 14

        for axis, result in zip(axes.reshape(-1), per_time_results):
            gt_log_probs = result["gt_log_probs"]
            model_log_probs = result["model_log_probs"]
            correlation = result["correlation"]

            axis.scatter(
                gt_log_probs.numpy(),
                model_log_probs.numpy(),
                s=4,
                alpha=0.2,
                rasterized=True,
            )
            min_val = min(gt_log_probs.min().item(), model_log_probs.min().item())
            max_val = max(gt_log_probs.max().item(), model_log_probs.max().item())
            axis.set_xlim(min_val, max_val)
            axis.set_ylim(min_val, max_val)
            axis.set_box_aspect(1)
            axis.plot(
                [min_val, max_val],
                [min_val, max_val],
                color="tab:red",
                linewidth=2.0,
            )
            axis.set_xlabel("Ground truth log density", fontsize=label_fontsize)
            axis.set_ylabel("Estimated log density", fontsize=label_fontsize)
            axis.set_title(
                f"t={result['time']:.2f}, corr={correlation:.4f}",
                fontsize=title_fontsize,
            )
            axis.tick_params(axis="both", which="major", labelsize=tick_fontsize)

        for axis in axes.reshape(-1)[num_times:]:
            axis.axis("off")

        scatter_path = os.path.join(eval_dir, "test_log_density_scatter_by_time.pdf")
        plt.savefig(scatter_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)

        for result in per_time_results:
            logging.info(
                "Estimated test log-density correlation at t=%.2f: %.6f",
                result["time"],
                result["correlation"],
            )
        logging.info("Saved time-grid log-density summary to %s", summary_path)
        logging.info(
            "Saved time-grid test log-density scatter plot to %s", scatter_path
        )

    if config.eval.sample:
        velocity_wrapper = VelocityWrapper(
            model=model,
            prob_path=get_prob_path(
                config=config,
                device=config.device,
                data_statistics=scaled_data_statistics,
                fm_model=None,
            ),
        )

        samples = euler_sample(
            velocity_wrapper=velocity_wrapper,
            num_samples=64,
            batch_size=64,
            data_shape=config.data.shape,
            num_steps=1000,
            device=config.device,
        )

        samples = samples.detach().cpu()
        samples = inverse_scaler(samples)

        save_image(samples, os.path.join(eval_dir, "samples.png"))
