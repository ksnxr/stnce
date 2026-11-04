import math
import torch
import logging
from . import data_utils
from torch.distributions import (
    Categorical,
    MultivariateNormal,
    MixtureSameFamily,
)
from .default import register_metric
from utils import random_orthogonal, softplus_inverse


@data_utils.register_data(name="param_gaussian")
class ParameterizedGaussian:
    def __init__(self, config, workdir):
        prefix = "param_gaussian"
        self.workdir = workdir
        self.dim = config.data.dim
        self.device = config.device
        self.dtype = config.dtype

        self.save_path = (
            f"data_cache/{prefix}_{self.dim}_{config.data.num_train_data}.pt"
        )

        self.train_set_path = (
            f"train_sets/{prefix}_{self.dim}_{config.data.num_train_data}.pt"
        )

        self.dist_mean = torch.zeros(self.dim, device=self.device, dtype=self.dtype)
        self.precision = None
        self.factor = None
        self.dist = None
        self.log_z = None

        self.plot_lims = (-3.0, 3.0)

    def get_cache(self):
        # random factors from covariances
        vector = random_orthogonal(1, self.dim, device=self.device, dtype=self.dtype)
        scale = 1.0 / (
            0.1 + 0.9 * torch.rand((1, self.dim), device=self.device, dtype=self.dtype)
        )
        precision = vector @ torch.diag_embed(scale) @ vector.transpose(-1, -2)
        cache = {"precision": precision.squeeze(dim=0)}
        return cache

    def load_cache(self, cache):
        self.precision = cache["precision"]
        self.factor = torch.linalg.cholesky(self.precision)
        self.log_z = -self.dim / 2 * math.log(2 * math.pi) + 1 / 2 * torch.logdet(
            self.precision
        )

        self.dist = MultivariateNormal(
            loc=self.dist_mean, precision_matrix=self.precision
        )

    def get_gt_model_state_dict(self, model):
        state_dict = dict()
        indices = torch.tril_indices(self.dim, self.dim, offset=-1)
        cholesky_offdiag = self.factor[indices[0], indices[1]]
        cholesky_diag_unconstrained = softplus_inverse(
            torch.diagonal(self.factor) - model.eps
        )

        state_dict["cholesky_offdiag"] = cholesky_offdiag.clone()
        state_dict["cholesky_diag_unconstrained"] = cholesky_diag_unconstrained.clone()
        state_dict["log_z"] = self.log_z.clone()
        return state_dict

    def sample(self, num_samples):
        return self.dist.sample(sample_shape=(num_samples,))

    def log_prob(self, x):
        return self.dist.log_prob(x).unsqueeze(dim=-1)

    @torch.no_grad()
    def evaluate(
        self,
        step,
        model,
        metrics,
        eval_log_z=False,
    ):
        logging.info(f"Step {step} evaluating:")

        # Cholesky factor of precision
        model_factor = model.get_factor()
        register_metric(metrics, step, "model_factor", model_factor.cpu().numpy())
        factor_error = torch.sum(torch.square(self.factor - model_factor)).item()
        register_metric(metrics, step, "factor_error", factor_error)

        logging.info(f"True factor: {self.factor}")
        logging.info(f"Model factor: {model_factor}")

        # log_z error
        if eval_log_z:
            model_log_z = model.log_z
            register_metric(metrics, step, "model_log_z", model_log_z.cpu().item())
            log_z_error = torch.square(self.log_z - model.log_z).item()
            register_metric(metrics, step, "log_z_error", log_z_error)

        # total error, only factor error
        register_metric(metrics, step, "total_error", factor_error)

        logging.info(f"Total error: {factor_error}")

        logging.info("")
        return metrics


@data_utils.register_data(name="param_two_gaussians")
class ParameterizedTwoGaussians:
    def __init__(self, config, workdir):
        prefix = "param_two_gaussians"
        self.workdir = workdir
        self.dim = config.data.dim
        self.device = config.device
        self.dtype = config.dtype

        self.save_path = (
            f"data_cache/{prefix}_{self.dim}_{config.data.num_train_data}.pt"
        )

        self.train_set_path = (
            f"train_sets/{prefix}_{self.dim}_{config.data.num_train_data}.pt"
        )

        self.dist_means = torch.stack(
            [
                torch.full((self.dim,), 1.0, device=self.device, dtype=self.dtype),
                torch.full((self.dim,), -1.0, device=self.device, dtype=self.dtype),
            ],
            dim=0,
        )
        self.factors = None
        self.weights = None
        self.dist = None

        self.plot_lims = (-2.0, 2.0)

    def get_cache(self):
        # random factors from covariances
        vectors = random_orthogonal(2, self.dim, device=self.device, dtype=self.dtype)
        scales = 0.1 + 0.9 * torch.rand(
            (2, self.dim), device=self.device, dtype=self.dtype
        )
        covs = vectors @ torch.diag_embed(scales) @ vectors.transpose(-1, -2)
        factors = torch.linalg.cholesky(covs)
        # prob1 is between 0.25 and 0.75
        prob1 = 0.25 + 0.5 * torch.rand(1, device=self.device, dtype=self.dtype).item()
        cache = {"factors": factors, "prob1": prob1}
        return cache

    def load_cache(self, cache):
        self.factors = cache["factors"]
        prob1 = cache["prob1"]
        self.logit1 = math.log(prob1 / (1.0 - prob1))

        self.weights = torch.tensor(
            [1.0 - prob1, prob1], device=self.device, dtype=self.dtype
        )
        mix = Categorical(probs=self.weights)
        comps = MultivariateNormal(loc=self.dist_means, scale_tril=self.factors)
        self.dist = MixtureSameFamily(mix, comps, validate_args=False)

    def get_gt_model_state_dict(self, model):
        state_dict = dict()

        indices = torch.tril_indices(self.dim, self.dim, offset=-1)
        cholesky_offdiag = self.factors[:, indices[0], indices[1]]
        cholesky_diag_unconstrained = softplus_inverse(
            torch.diagonal(self.factors, dim1=1, dim2=2) - model.eps
        )

        state_dict["means"] = self.dist_means.clone()
        state_dict["cholesky_offdiag"] = cholesky_offdiag.clone()
        state_dict["cholesky_diag_unconstrained"] = cholesky_diag_unconstrained.clone()
        state_dict["logit"] = self.logit1.clone()
        return state_dict

    def sample(self, num_samples):
        return self.dist.sample((num_samples,))

    def log_prob(self, x):
        return self.dist.log_prob(x).unsqueeze(dim=-1)

    @torch.no_grad()
    def evaluate(
        self,
        step,
        model,
        metrics,
    ):
        logging.info(f"Step {step} evaluating:")

        cost_id = torch.sum(
            torch.square(self.dist_means[0] - model.means[0])
        ) + torch.sum(torch.square(self.dist_means[1] - model.means[1]))
        cost_swap = torch.sum(
            torch.square(self.dist_means[0] - model.means[1])
        ) + torch.sum(torch.square(self.dist_means[1] - model.means[0]))
        if cost_id <= cost_swap:
            indices = torch.tensor([0, 1], device=self.device, dtype=torch.long)
        else:
            indices = torch.tensor([1, 0], device=self.device, dtype=torch.long)

        # means
        model_means = model.get_means()[indices]
        register_metric(metrics, step, "model_means", model_means.cpu().numpy())
        means_error = torch.sum(torch.square(self.dist_means - model_means)).item()
        register_metric(metrics, step, "means_error", means_error)

        logging.info(f"True means: {self.dist_means}")
        logging.info(f"Model means: {model_means}")

        # Cholesky factors of covariances
        model_factors = model.get_factors()[indices]
        register_metric(metrics, step, "model_factors", model_factors.cpu().numpy())
        factors_error = torch.sum(torch.square(self.factors - model_factors)).item()
        register_metric(metrics, step, "factors_error", factors_error)

        logging.info(f"True factors: {self.factors}")
        logging.info(f"Model factors: {model_factors}")

        # weights
        model_logit1 = model.logit.cpu().item()
        register_metric(metrics, step, "model_logit1", model_logit1)
        logit1_error = (self.logit1 - model_logit1) ** 2
        register_metric(metrics, step, "logit1_error", logit1_error)

        logging.info(f"True logit1: {self.logit1}")
        logging.info(f"Model logit1: {model_logit1}")

        total_error = means_error + factors_error + logit1_error
        register_metric(metrics, step, "total_error", total_error)

        logging.info(f"Total error: {total_error}")

        logging.info("")
        return metrics
