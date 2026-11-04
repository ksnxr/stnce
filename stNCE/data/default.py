import os
import torch
import logging
import matplotlib.pyplot as plt
from torch.distributions import Distribution
from utils import (
    estimate_log_const_importance,
    estimate_log_const_importance_from_dataloader,
    register_metric,
    update_weighted_running_mean,
)


def generate_plot(
    plot_lims,
    model_energy_fn,
    log_data_fn,
    device,
    workdir,
    name,
    batch_size,
    log_constant=0.0,
):

    xs = torch.linspace(plot_lims[0], plot_lims[1], 100, device=device)
    ys = torch.linspace(plot_lims[0], plot_lims[1], 100, device=device)
    xs, ys = torch.meshgrid(xs, ys, indexing="ij")
    coords = torch.stack([xs, ys], dim=-1)
    coords_flat = coords.view((-1, 2))

    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(coords_flat),
        batch_size=batch_size,
        shuffle=False,
    )
    model_lps = []
    data_lps = []
    for batch in dataloader:
        batch_coords = batch[0].to(device)
        model_lps.append(-model_energy_fn(batch_coords))
        data_lps.append(log_data_fn(batch_coords))

    model_lps = torch.cat(model_lps, dim=0).view((100, 100)) - log_constant
    data_lps = torch.cat(data_lps, dim=0).view((100, 100))

    plt.rcParams.update({"text.usetex": True, "font.size": 15})

    plt.figure(figsize=(5, 5))
    plt.imshow(
        torch.exp(model_lps.detach()).cpu().numpy(),
        extent=[
            plot_lims[0],
            plot_lims[1],
            plot_lims[0],
            plot_lims[1],
        ],
        origin="lower",
        cmap="viridis",
        aspect="equal",
        interpolation="bilinear",
    )
    plt.xlim(plot_lims)
    plt.ylim(plot_lims)

    plt.tight_layout()
    plt.savefig(f"{workdir}/{name}_plot.pdf", bbox_inches="tight")
    plt.close()


class DefaultDistribution(Distribution):
    def __init__(self, config, workdir, prefix):
        self.workdir = workdir
        self.train_set_path = f"train_sets/{prefix}_{config.data.dim}_{config.seed}_{config.data.num_train_data}.pt"
        self.val_set_path = f"val_sets/{prefix}_{config.data.dim}_{config.seed}_{config.data.num_eval_data}.pt"
        self.test_set_path = f"test_sets/{prefix}_{config.data.dim}_{config.seed}_{config.data.num_eval_data}.pt"
        self.is_set_path = f"is_sets/{prefix}_{config.data.dim}_{config.seed}_{config.eval.is_num_samples}.pt"
        self.val_samples_dict = None
        self.test_samples_dict = None
        self.val_true_log_probs = None
        self.test_true_log_probs = None
        self.val_other_true_log_probs = None
        self.test_other_true_log_probs = None

    def load_eval_sets(self):
        if self.val_samples_dict is None:
            self.val_samples_dict = torch.load(self.val_set_path)
        if self.test_samples_dict is None:
            self.test_samples_dict = torch.load(self.test_set_path)

    def _compute_true_log_probs(self, samples, batch_size):
        samples_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(samples),
            batch_size=batch_size,
            shuffle=False,
        )
        true_log_probs = []
        for batch in samples_dataloader:
            batch_data = batch[0].to(device=self.device)
            true_log_probs.append(self.log_prob(batch_data))
        return torch.cat(true_log_probs, dim=0)

    def load_eval_log_probs(self, batch_size):
        self.load_eval_sets()
        if self.val_true_log_probs is None:
            self.val_true_log_probs = self._compute_true_log_probs(
                self.val_samples_dict["samples"], batch_size
            )
        if self.test_true_log_probs is None:
            self.test_true_log_probs = self._compute_true_log_probs(
                self.test_samples_dict["samples"], batch_size
            )
        if self.val_other_true_log_probs is None:
            self.val_other_true_log_probs = self._compute_true_log_probs(
                self.val_samples_dict["other_samples"], batch_size
            )
        if self.test_other_true_log_probs is None:
            self.test_other_true_log_probs = self._compute_true_log_probs(
                self.test_samples_dict["other_samples"], batch_size
            )

    @torch.no_grad()
    def evaluate(
        self,
        step,
        model,
        batch_size,
        build_noise_fn,
        eval_loss_fn,
        metrics,
        get_ebm_fn,
        is_num_samples=100000,
        is_dataloader=None,
        save_plot=True,
    ):
        logging.info(f"Step {step} evaluating:")

        ebm_model = get_ebm_fn(model)

        if is_dataloader is None:
            logging.info("Warning: no is_dataloader passed. Generating on the fly.")
            log_constant = estimate_log_const_importance(
                energy_fn=ebm_model,
                proposal_dist=self.dist,
                num_samples=is_num_samples,
                batch_size=batch_size,
                return_diagnostics=False,
            )
        else:
            log_constant = estimate_log_const_importance_from_dataloader(
                energy_fn=ebm_model,
                proposal_log_prob_fn=self.log_prob,
                is_dataloader=is_dataloader,
                device=self.device,
            )
        register_metric(metrics, step, "log_constant", log_constant)

        self.load_eval_sets()
        self.load_eval_log_probs(batch_size)

        for set_name, samples_dict, true_log_probs_all, other_true_log_probs_all in [
            [
                "val",
                self.val_samples_dict,
                self.val_true_log_probs,
                self.val_other_true_log_probs,
            ],
            [
                "test",
                self.test_samples_dict,
                self.test_true_log_probs,
                self.test_other_true_log_probs,
            ],
        ]:
            assert (
                samples_dict["samples"].shape[0]
                == samples_dict["other_samples"].shape[0]
            )
            samples_dataloader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(samples_dict["samples"]),
                batch_size=batch_size,
                shuffle=False,
            )
            other_samples_dataloader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(samples_dict["other_samples"]),
                batch_size=batch_size,
                shuffle=False,
            )
            avg_mse = torch.zeros((), device=self.device, dtype=torch.float64)
            avg_norm_mse = torch.zeros((), device=self.device, dtype=torch.float64)
            avg_norm_nll = torch.zeros((), device=self.device, dtype=torch.float64)
            avg_ratio_loss = torch.zeros((), device=self.device, dtype=torch.float64)
            num_samples = torch.zeros((), device=self.device, dtype=torch.float64)
            if eval_loss_fn is not None:
                avg_loss = torch.zeros((), device=self.device, dtype=torch.float64)
                num_loss_batches = torch.zeros(
                    (), device=self.device, dtype=torch.float64
                )

            sample_index = 0
            for batch, other_batch in zip(samples_dataloader, other_samples_dataloader):
                batch_data = batch[0].to(device=self.device)
                other_batch_data = other_batch[0].to(device=self.device)
                next_sample_index = sample_index + batch_data.shape[0]
                true_log_probs = true_log_probs_all[
                    sample_index:next_sample_index
                ].reshape(-1)
                model_log_probs = (-ebm_model(batch_data)).reshape(-1)
                other_true_log_probs = other_true_log_probs_all[
                    sample_index:next_sample_index
                ].reshape(-1)
                other_model_log_probs = (-ebm_model(other_batch_data)).reshape(-1)
                batch_weight = batch_data.shape[0]
                prev_num_samples = num_samples
                sample_index = next_sample_index

                mse_diff = (true_log_probs - model_log_probs).to(torch.float64)
                batch_mse = torch.mean(mse_diff * mse_diff)
                normalized_model_log_probs = model_log_probs - log_constant
                norm_mse_diff = (true_log_probs - normalized_model_log_probs).to(
                    torch.float64
                )
                batch_norm_mse = torch.mean(norm_mse_diff * norm_mse_diff)
                batch_norm_nll = -torch.mean(
                    normalized_model_log_probs.to(torch.float64)
                )
                ratio_diff = (
                    (other_true_log_probs - true_log_probs)
                    - (other_model_log_probs - model_log_probs)
                ).to(torch.float64)
                batch_ratio_loss = torch.mean(ratio_diff * ratio_diff)

                avg_mse, num_samples = update_weighted_running_mean(
                    avg_mse, num_samples, batch_mse, batch_weight
                )
                avg_norm_mse, _ = update_weighted_running_mean(
                    avg_norm_mse, prev_num_samples, batch_norm_mse, batch_weight
                )
                avg_norm_nll, _ = update_weighted_running_mean(
                    avg_norm_nll, prev_num_samples, batch_norm_nll, batch_weight
                )
                avg_ratio_loss, _ = update_weighted_running_mean(
                    avg_ratio_loss, prev_num_samples, batch_ratio_loss, batch_weight
                )

                if eval_loss_fn is not None:
                    if build_noise_fn is not None:
                        batch = build_noise_fn(
                            original_data=batch_data, model_energy_fn=ebm_model.model
                        )
                        losses = eval_loss_fn(model, batch)
                        if type(losses) == tuple:
                            batch_loss = losses[-1].detach()
                        else:
                            batch_loss = losses.detach()
                    else:
                        batch = batch_data
                        with torch.enable_grad():
                            losses = eval_loss_fn(model, batch)
                            if type(losses) == tuple:
                                batch_loss = losses[-1].detach()
                            else:
                                batch_loss = losses.detach()
                    avg_loss, num_loss_batches = update_weighted_running_mean(
                        avg_loss, num_loss_batches, batch_loss, 1.0
                    )

            if eval_loss_fn is not None:
                loss = avg_loss.item()

                metric_name = f"{set_name}_loss"
                if set_name == "val":
                    logging.info(f"{metric_name}: {loss:.3f}")
                register_metric(metrics, step, metric_name, loss)

            # mse
            mse = avg_mse.item()
            register_metric(metrics, step, f"{set_name}_mse", mse)

            # normalized mse
            normalized_mse = avg_norm_mse.item()
            register_metric(metrics, step, f"{set_name}_norm_mse", normalized_mse)

            normalized_nll = avg_norm_nll.item()
            register_metric(metrics, step, f"{set_name}_norm_nll", normalized_nll)

            ratio_loss = avg_ratio_loss.item()
            register_metric(metrics, step, f"{set_name}_real_ratio", ratio_loss)

        if (self.dim == 2) and save_plot:
            if not os.path.exists(f"{self.workdir}/true_plot.pdf"):
                generate_plot(
                    plot_lims=self.plot_lims,
                    model_energy_fn=lambda x: -self.dist.log_prob(x),
                    log_data_fn=self.dist.log_prob,
                    device=self.device,
                    workdir=self.workdir,
                    name="true",
                    batch_size=batch_size,
                    log_constant=0.0,
                )

            generate_plot(
                plot_lims=self.plot_lims,
                model_energy_fn=ebm_model,
                log_data_fn=self.dist.log_prob,
                device=self.device,
                workdir=self.workdir,
                name=step,
                batch_size=batch_size,
                log_constant=log_constant,
            )

        logging.info("")
        return metrics

    @torch.no_grad()
    def evaluate_fm(
        self,
        prob_path,
        step,
        model,
        batch_size,
        build_noise_fn,
        eval_loss_fn,
        metrics,
    ):
        logging.info(f"Step {step} evaluating:")

        self.load_eval_sets()

        gt_score_fn = prob_path.ground_truth_score

        for set_name, samples_dict in [
            ["val", self.val_samples_dict],
            ["test", self.test_samples_dict],
        ]:
            samples_dataloader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(samples_dict["samples"]),
                batch_size=batch_size,
                shuffle=False,
            )
            avg_loss = torch.zeros((), device=self.device, dtype=torch.float64)
            avg_score_error = torch.zeros((), device=self.device, dtype=torch.float64)
            num_samples = torch.zeros((), device=self.device, dtype=torch.float64)
            for batch in samples_dataloader:
                batch_data = batch[0].to(device=self.device)
                batch_weight = batch_data.shape[0]
                batch = build_noise_fn(original_data=batch_data)

                losses = eval_loss_fn(model, batch)
                if type(losses) == tuple:
                    batch_loss = losses[-1].detach()
                else:
                    batch_loss = losses.detach()
                prev_num_samples = num_samples
                avg_loss, num_samples = update_weighted_running_mean(
                    avg_loss, num_samples, batch_loss, batch_weight
                )

                ts = prob_path.sample_t(batch_data.shape[0], device=batch_data.device)
                xs = prob_path.sample(z=batch_data, t=ts)
                model_scores = prob_path.score_from_v(vs=model(xs, ts), xs=xs, t=ts)
                gt_scores = gt_score_fn(xs, ts)

                batch_score_error = torch.mean(
                    torch.mean(
                        torch.square((model_scores - gt_scores).to(torch.float64)),
                        dim=-1,
                    )
                    * torch.square(prob_path.beta(ts).to(torch.float64))
                )
                avg_score_error, _ = update_weighted_running_mean(
                    avg_score_error, prev_num_samples, batch_score_error, batch_weight
                )

            loss = avg_loss.item()
            score_error = avg_score_error.item()

            metric_name = f"{set_name}_loss"
            if set_name == "val":
                logging.info(f"{metric_name}: {loss:.3f}")
            register_metric(metrics, step, metric_name, loss)

            metric_name = f"{set_name}_score"
            if set_name == "val":
                logging.info(f"{metric_name}: {score_error:.3f}")
            register_metric(metrics, step, metric_name, score_error)

        logging.info("")
        return metrics
