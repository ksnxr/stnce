import math
import os
import time
import shutil
import torch
import logging
import losses_lib

import pickle


# for registrations
from models import toy_models
from data import toy_data

from models import model_utils as mutils
from models.ema import ExponentialMovingAverage
from data import data_utils as dutils
from absl import flags
from estimator_utils import build_estimator_fns

from utils import seed_all
from prob_paths_lib import get_prob_path

FLAGS = flags.FLAGS


def _synchronized_perf_counter(device):
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    return time.perf_counter()


def _sample_dataset_to_cpu(dataset, num_samples, batch_size):
    num_samples = int(num_samples)
    batch_size = int(batch_size)
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    samples = None
    offset = 0
    with torch.no_grad():
        while offset < num_samples:
            current_batch_size = min(batch_size, num_samples - offset)
            batch = dataset.sample(current_batch_size).detach().cpu()
            if batch.shape[0] != current_batch_size:
                raise ValueError(
                    "Dataset sample returned an unexpected leading dimension: "
                    f"expected {current_batch_size}, got {batch.shape[0]}."
                )
            if samples is None:
                samples = torch.empty(
                    (num_samples, *batch.shape[1:]), dtype=batch.dtype
                )
            samples[offset : offset + current_batch_size] = batch
            offset += current_batch_size

    return samples


def _load_is_dataloader(dataset, batch_size):
    is_set_path = getattr(dataset, "is_set_path", None)
    if is_set_path is None:
        return None
    if not os.path.exists(is_set_path):
        logging.warning(
            f"IS set cache {is_set_path} does not exist; falling back to "
            "on-the-fly importance sampling."
        )
        return None

    cached_set = torch.load(is_set_path, map_location="cpu")
    if isinstance(cached_set, dict):
        samples = cached_set["samples"]
    else:
        samples = cached_set
    is_dataset = torch.utils.data.TensorDataset(samples)
    return torch.utils.data.DataLoader(
        is_dataset,
        batch_size=batch_size,
        shuffle=False,
    )


def train(config, workdir):
    start_time = _synchronized_perf_counter(config.device)
    seed_all(config.seed)

    dataset = dutils.create_data(config, workdir=workdir)
    if config.training.setup:
        return  # Only setup, no training

    estimator = config.estim.name
    reg_estimator = config.estim.reg_estimator
    num_train_steps = config.training.num_iters
    num_noise_classes = config.estim.num_noise_classes
    device = config.device
    eval_batch_size = config.eval.batch_size
    eval_is_num_samples = config.eval.is_num_samples

    if num_train_steps == 0:
        # save train set, val set, test set and exit
        for set_dir in ["train_sets", "val_sets", "test_sets", "is_sets"]:
            os.makedirs(set_dir, exist_ok=True)

        train_data_tensor = _sample_dataset_to_cpu(
            dataset,
            num_samples=config.data.num_train_data,
            batch_size=eval_batch_size,
        )
        if os.path.exists(dataset.train_set_path):
            logging.info(
                f"Removing existing {dataset.train_set_path} for reinitialization."
            )
            os.remove(dataset.train_set_path)

        torch.save(train_data_tensor, dataset.train_set_path)
        del train_data_tensor

        for set_path in ["val_sets", "test_sets"]:
            samples_dict = dict()

            samples = _sample_dataset_to_cpu(
                dataset,
                num_samples=config.data.num_eval_data,
                batch_size=eval_batch_size,
            )
            samples_dict["samples"] = samples

            other_samples = _sample_dataset_to_cpu(
                dataset,
                num_samples=config.data.num_eval_data,
                batch_size=eval_batch_size,
            )
            samples_dict["other_samples"] = other_samples

            if set_path == "val_sets":
                save_path = dataset.val_set_path
            elif set_path == "test_sets":
                save_path = dataset.test_set_path
            else:
                raise NotImplementedError
            if os.path.exists(save_path):
                logging.info(f"Removing existing {save_path} for reinitialization.")
                os.remove(save_path)

            torch.save(samples_dict, save_path)
            del samples_dict, samples, other_samples

        if hasattr(dataset, "is_set_path"):
            is_samples = _sample_dataset_to_cpu(
                dataset,
                num_samples=eval_is_num_samples,
                batch_size=eval_batch_size,
            )
            if os.path.exists(dataset.is_set_path):
                logging.info(
                    f"Removing existing {dataset.is_set_path} for reinitialization."
                )
                os.remove(dataset.is_set_path)
            torch.save({"samples": is_samples}, dataset.is_set_path)
            del is_samples
        return

    train_data_tensor = torch.load(dataset.train_set_path).to(device=device)
    is_dataloader = _load_is_dataloader(dataset, eval_batch_size)

    model = mutils.create_model(config).to(device=device)

    if config.model.name.startswith("precond_"):
        specified_log_const = 0.5 * math.log(2 * math.pi) * config.data.dim
    else:
        specified_log_const = None

    param_count = sum(p.numel() for p in model.parameters())
    logging.info(f"Model has {param_count} params")

    data_statistics = dataset.get_data_statistics()

    if config.model.fm_model_path == "none":
        fm_model = None
    else:
        fm_config = config.copy_and_resolve_references()
        fm_config.model.out_dim = "data"
        fm_model = mutils.create_model(fm_config).to(device=device)
        fm_model.load_state_dict(
            torch.load(config.model.fm_model_path, map_location=device)["model"]
        )
        fm_model.eval()

    if config.estim.num_noise_classes == float("inf"):
        prob_path = get_prob_path(
            config=config,
            device=device,
            data_statistics=data_statistics,
            fm_model=fm_model,
        )
    else:
        raise NotImplementedError

    main_build_noise_fn, reg_build_noise_fn, build_noise_fn, get_ebm, reg_estimator = (
        build_estimator_fns(
            config=config,
            prob_path=prob_path,
            dim=dataset.dim,
            data_shape=[dataset.dim],
            specified_log_const=specified_log_const,
            force_fm_reg_estimator_none=False,
        )
    )

    if (estimator == "fm") and (config.estim.param == "velocity"):

        def evaluate_fn(
            current_build_noise_fn, current_eval_loss_fn, step, model, metrics
        ):
            return dataset.evaluate_fm(
                prob_path=prob_path,
                step=step,
                model=model,
                batch_size=eval_batch_size,
                build_noise_fn=current_build_noise_fn,
                eval_loss_fn=current_eval_loss_fn,
                metrics=metrics,
            )

    else:

        def evaluate_fn(
            current_build_noise_fn, current_eval_loss_fn, step, model, metrics
        ):
            return dataset.evaluate(
                step=step,
                model=model,
                batch_size=eval_batch_size,
                build_noise_fn=current_build_noise_fn,
                eval_loss_fn=current_eval_loss_fn,
                metrics=metrics,
                get_ebm_fn=get_ebm,
                is_num_samples=eval_is_num_samples,
                is_dataloader=is_dataloader,
            )

    optimizer = losses_lib.get_optimizer(config, model.parameters())
    initial_step = 1

    use_ema = config.model.use_ema
    if use_ema:
        ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
        state = dict(optimizer=optimizer, model=model, ema=ema)
    else:
        state = dict(optimizer=optimizer, model=model)

    train_dataset = torch.utils.data.TensorDataset(train_data_tensor)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        drop_last=True,
    )
    train_iter = iter(train_dataloader)

    train_step_fn = losses_lib.get_step_fn(
        estimator=estimator,
        num_noise_classes=num_noise_classes,
        num_cla_classes=config.estim.num_cla_classes,
        train=True,
        reg_estimator=reg_estimator,
        nce_nu=config.estim.nce_nu,
        cnce_nu=config.estim.cnce_nu,
        param=config.estim.param,
        use_ema=use_ema,
        main_weight=config.estim.main_weight,
        reg_weight=config.estim.reg_weight,
        combine_type=config.estim.combine_type,
        mixture_loss=getattr(config.estim, "mixture_loss", False),
        rne=config.estim.rne,
        prob_path=prob_path,
    )
    eval_loss_fn = losses_lib.get_loss_fn(
        estimator=estimator,
        reg_estimator=reg_estimator,
        num_noise_classes=num_noise_classes,
        num_cla_classes=config.estim.num_cla_classes,
        param=config.estim.param,
        nce_nu=config.estim.nce_nu,
        cnce_nu=config.estim.cnce_nu,
        main_weight=config.estim.main_weight,
        reg_weight=config.estim.reg_weight,
        combine_type=config.estim.combine_type,
        mixture_loss=getattr(config.estim, "mixture_loss", False),
        rne=config.estim.rne,
        prob_path=prob_path,
    )

    eval_freq = config.eval.eval_freq

    metrics = dict()

    save_checkpoint = config.training.save_checkpoint
    selected_metric = config.eval.selected_metric
    selected_metrics = [
        "val_mse",
        "val_norm_mse",
        "val_loss",
        "val_real_ratio",
        "val_norm_nll",
    ]
    best_metric_values = {metric_name: float("inf") for metric_name in selected_metrics}
    selected_checkpoint_paths = {
        "val_mse": os.path.join(workdir, "best_val_mse_model.pt"),
        "val_norm_mse": os.path.join(workdir, "best_val_norm_mse_model.pt"),
        "val_loss": os.path.join(workdir, "best_val_loss_model.pt"),
        "val_real_ratio": os.path.join(workdir, "best_val_real_ratio_model.pt"),
        "val_norm_nll": os.path.join(workdir, "best_val_norm_nll_model.pt"),
    }
    train_time_between_eval_steps = dict()

    model.train()
    last_eval_end_time = _synchronized_perf_counter(device)

    for step in range(initial_step, num_train_steps + 1):
        try:
            data = next(train_iter)[0].to(device=device)
        except StopIteration:
            train_iter = iter(train_dataloader)
            data = next(train_iter)[0].to(device=device)

        train_step_fn(state, data, build_noise_fn)

        if step % eval_freq == 0:
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
                metrics = evaluate_fn(
                    current_build_noise_fn=build_noise_fn,
                    current_eval_loss_fn=eval_loss_fn,
                    step=step,
                    model=model,
                    metrics=metrics,
                )
                if use_ema:
                    ema.restore(model.parameters())
                if was_training:
                    model.train()

                for metric_name in selected_metrics:
                    current_metric = metrics[metric_name][step]
                    if current_metric < best_metric_values[metric_name]:
                        best_metric_values[metric_name] = current_metric
                        logging.info(
                            f"New best {metric_name} found at step {step} with {current_metric:.4f}"
                        )

                        if save_checkpoint:
                            checkpoint_path = selected_checkpoint_paths[metric_name]
                            save_state = {
                                "step": step,
                                "model_state_dict": state["model"].state_dict(),
                                "optimizer_state_dict": state["optimizer"].state_dict(),
                            }
                            if use_ema:
                                save_state["ema_state_dict"] = state["ema"].state_dict()
                            torch.save(save_state, checkpoint_path)
                            logging.info(f"Saved checkpoint to {checkpoint_path}")

            last_eval_end_time = _synchronized_perf_counter(device)

    selected_metric_values = metrics[selected_metric]
    metrics["best"] = dict()
    best_step = min(selected_metric_values, key=selected_metric_values.get)
    logging.info(
        f"Best {selected_metric} at step {best_step}: {selected_metric_values[best_step]:.6f}"
    )

    if os.path.exists(f"{workdir}/{best_step}_plot.pdf"):
        shutil.copy(f"{workdir}/{best_step}_plot.pdf", f"{workdir}/best_plot.pdf")
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
            if config.print_test:
                test_metric_values = metrics[f"test_{raw_metric_name}"]
                logging.info(
                    f"{raw_metric_name} test set result: {test_metric_values[best_step]:.3f} at {best_step}"
                )
                metrics["best"][f"test_{raw_metric_name}"] = test_metric_values[
                    best_step
                ]

    end_time = _synchronized_perf_counter(device)
    times = [start_time, end_time, end_time - start_time]
    metrics["times"] = times
    metrics["train_time_between_eval_steps"] = train_time_between_eval_steps
    with open(f"{workdir}/metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)


def evaluate(config, workdir, eval_folder):
    raise NotImplementedError
