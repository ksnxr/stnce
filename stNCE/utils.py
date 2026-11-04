import math
import torch
import torch.nn as nn
import logging
import random
import numpy as np
from torch.distributions import Normal, Categorical, Independent, MixtureSameFamily
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from sampling.utils import ModelWrapper
from sampling.ode_solver import ODESolver
from analytics import gaussian_score, gaussian_mixture_score


# diffusion scaler


def scaler(data):
    return 2.0 * data - 1.0


def scaler_data_statistics(data_statistics):
    data_statistics["means"] = scaler(data_statistics["means"])
    data_statistics["std"] = 2.0 * data_statistics["std"]
    return data_statistics


def scaler_log_det(data):
    dim = data[0].numel()
    return torch.full((data.shape[0],), math.log(2.0) * dim, device=data.device)


def inverse_scaler(data):
    return data / 2.0 + 0.5


def inverse_scaler_log_det(data):
    dim = data[0].numel()
    return torch.full((data.shape[0],), math.log(0.5) * dim, device=data.device)


def log_info(message, accelerator=None):
    if (accelerator is None) or accelerator.is_main_process:
        logging.info(message)


@torch.no_grad()
def estimate_log_const_importance(
    energy_fn,  # callable: x -> energy (batch,)
    proposal_dist,  # torch.distributions.Distribution with .sample, .log_prob
    num_samples: int = 100_000,
    batch_size: int = 4096,
    device: torch.device = None,
    return_diagnostics: bool = False,
):
    """
    Importance sampling estimate of log normalizer (log Z) for an EBM.

    Args:
      energy_fn: callable mapping x -> E_theta(x). Should accept (B, D) and return (B,)
      proposal_dist: torch.distributions object over R^D with .sample, .log_prob
      num_samples: total proposal samples
      batch_size: chunk size to avoid OOM
      device: optional torch.device
      return_diagnostics: if True, also returns ESS, SE, etc.

    Returns:
      logZ_hat (tensor scalar), and optionally a dict of diagnostics.
    """
    if device is None:
        # try to put on the proposal's parameters device if available
        try:
            device = next(proposal_dist.parameters()).device
        except Exception:
            device = torch.device("cpu")

    # Running log-sums for numerical stability
    lse_w = torch.tensor(-float("inf"), device=device)  # log(sum w)
    if return_diagnostics:
        lse_w2 = torch.tensor(-float("inf"), device=device)  # log(sum w^2)

    remaining = num_samples
    while remaining > 0:
        b = min(batch_size, remaining)
        # sample from q
        x = proposal_dist.sample((b,)).to(device)
        # log q(x)
        log_q = proposal_dist.log_prob(x)
        if log_q.ndim > 1:
            log_q = log_q.view(-1)

        # energy
        lps = -energy_fn(x)
        if lps.ndim > 1:
            lps = lps.view(-1)

        # log-weights: log w = -E(x) - log q(x)
        logw = lps - log_q

        # accumulate log(sum w) and log(sum w^2) stably
        lse_w = torch.logaddexp(lse_w, torch.logsumexp(logw, dim=0))
        if return_diagnostics:
            lse_w2 = torch.logaddexp(lse_w2, torch.logsumexp(2.0 * logw, dim=0))

        remaining -= b

    # log Z hat = log( sum w / N )
    logZ_hat = lse_w - math.log(num_samples)

    if not return_diagnostics:
        return logZ_hat

    # Diagnostics (work in log-space, exponentiate only scalars at the end)
    sum_w = torch.exp(lse_w)
    sum_w2 = torch.exp(lse_w2)

    Z_hat = torch.clamp(sum_w / num_samples, min=1e-300)
    # Var(Z_hat) ≈ (E_q[w^2] - Z^2)/N ; estimate E_q[w^2] by sample mean
    var_Z_hat = (sum_w2 / num_samples - Z_hat**2) / num_samples
    var_Z_hat = torch.clamp(var_Z_hat, min=0.0)
    # Delta method: Var(log Z_hat) ≈ Var(Z_hat) / Z_hat^2
    se_logZ = torch.sqrt(var_Z_hat) / Z_hat

    # Effective sample size
    ess = (sum_w**2 / sum_w2).clamp(max=float(num_samples))

    diags = {
        "Z_hat": Z_hat.item(),
        "se_logZ": se_logZ.item(),
        "ESS": ess.item(),
        "ESS_frac": (ess / num_samples).item(),
        "num_samples": int(num_samples),
    }
    return logZ_hat, diags


@torch.no_grad()
def estimate_log_const_importance_from_dataloader(
    energy_fn,
    proposal_log_prob_fn,
    is_dataloader,
    device,
    accelerator=None,
):
    local_lse_w = torch.tensor(-float("inf"), device=device, dtype=torch.float64)
    local_num_samples = torch.zeros((), device=device, dtype=torch.float64)

    for batch in is_dataloader:
        batch_data = batch[0].to(device=device)
        log_q = proposal_log_prob_fn(batch_data)
        if log_q.ndim > 1:
            log_q = log_q.view(-1)
        lps = -energy_fn(batch_data)
        if lps.ndim > 1:
            lps = lps.view(-1)

        logw = lps.to(dtype=torch.float64) - log_q.to(dtype=torch.float64)
        local_lse_w = torch.logaddexp(local_lse_w, torch.logsumexp(logw, dim=0))
        local_num_samples += batch_data.shape[0]

    if accelerator is not None:
        gathered_stats = accelerator.gather(
            torch.stack([local_lse_w, local_num_samples]).unsqueeze(0)
        )
        valid_mask = gathered_stats[:, 1] > 0
        if not torch.any(valid_mask):
            raise ValueError("Importance-sampling dataloader must contain samples.")
        local_lse_w = torch.logsumexp(gathered_stats[valid_mask, 0], dim=0)
        local_num_samples = gathered_stats[valid_mask, 1].sum()

    if local_num_samples.item() <= 0:
        raise ValueError("Importance-sampling dataloader must contain samples.")

    log_constant_tensor = local_lse_w - torch.log(local_num_samples)
    return float(log_constant_tensor.item())


@torch.no_grad()
def estimate_log_const_entropy(
    energy_fn,
    dataloader,
    dim,
    t_max,
    device=None,
):

    energies = []

    for batch in dataloader:
        x = batch[0].to(device)
        energies.append(energy_fn(x).cpu())

    energies = torch.cat(energies, dim=0)
    logZ_hat = (
        dim / 2 * math.log(2 * math.pi * math.e * t_max) - torch.mean(energies).item()
    )
    return logZ_hat


class MyGMM:
    def __init__(
        self, num_components, dim, mus, sigma, device, dtype, num_limits, weights=None
    ):
        if num_components > num_limits:
            self.num_components = num_limits
            selected = torch.randperm(num_components)[:num_limits]
            self.mus = mus[selected, :]
        else:
            self.num_components = num_components
            self.mus = mus
        self.dim = dim
        self.sigma = sigma
        self.device = device
        self.dtype = dtype
        self.num_limits = num_limits
        if weights is None:
            self.weights = torch.full(
                (self.num_components,),
                1.0 / self.num_components,
                device=device,
                dtype=dtype,
            )
        else:
            self.weights = weights

        self.log_sigma = math.log(sigma)
        self.log_num_components = math.log(self.num_components)
        self.const = -0.5 * self.dim * math.log(2.0 * math.pi)

    def raw_log_prob(self, x):
        # Vectorized log-density for a Gaussian mixture with means `mus`
        # and scalar std `sigma`. Assumes uniform component weights.

        # Broadcast to (num_data, num_components, dim)
        z = (x.unsqueeze(1) - self.mus.unsqueeze(0)) / self.sigma
        # per-dimension Normal log-pdf terms
        per_dim = -0.5 * z.pow(2) - self.log_sigma
        log_p_x_given_k = self.const + per_dim.sum(dim=-1)  # (num_data, num_components)

        # Uniform mixture weights => add -log K then log-sum-exp over components
        return torch.logsumexp(
            log_p_x_given_k - self.log_num_components, dim=1, keepdim=True
        )

    def log_prob(self, x):
        num_noise = x.shape[0]
        if num_noise > self.num_limits:
            all_log_probs = torch.zeros((num_noise, 1))
            for i in range(math.floor(num_noise / self.num_limits)):
                start = i * self.num_limits
                end = min((i + 1) * self.num_limits, num_noise)
                all_log_probs[start:end, :] = self.raw_log_prob(x[start:end, :])
            return all_log_probs
        else:
            return self.raw_log_prob(x)

    def sample(self, num_samples):
        comp_idx = torch.multinomial(
            self.weights, num_samples=num_samples, replacement=True
        )
        mus_selected = self.mus.index_select(0, comp_idx)
        eps = torch.randn((num_samples, self.dim), device=self.device, dtype=self.dtype)
        samples = mus_selected + self.sigma * eps
        return samples


class MyGaussian:
    def __init__(self, dim, std):
        self.dim = dim
        self.std = std
        self.var = self.std**2
        self.log_norm = -0.5 * self.dim * math.log(2 * math.pi) - dim * math.log(std)

    def log_prob(self, diff):
        log_prob = (
            self.log_norm
            - 0.5 * torch.sum(torch.square(diff), dim=-1, keepdim=True) / self.var
        )
        return log_prob

    def sample(self, x):
        return x + self.std * torch.randn_like(x)


def build_gmm(
    num_components, dim, mus, sigma, device, dtype=torch.float32, weights=None
):
    # build the gmm using pytorch version
    if weights is None:
        weights = torch.full(
            (num_components,), 1.0 / num_components, device=device, dtype=dtype
        )
    mix = Categorical(probs=weights, validate_args=False)
    comps = Independent(
        Normal(
            mus,
            torch.full((num_components, dim), sigma, device=device, dtype=dtype),
            validate_args=False,
        ),
        reinterpreted_batch_ndims=1,
        validate_args=False,
    )
    dist = MixtureSameFamily(mix, comps, validate_args=False)
    return dist


def seed_all(seed, exact=True):
    logging.info(f"Fixing random seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # do not use exact to speed up for imagenet
        if exact:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True


def seed_rank_rng(seed, process_index):
    rank_seed = int(seed) + int(process_index)
    logging.info(
        "Fixing rank-local torch RNG seed to %s for process index %s",
        rank_seed,
        process_index,
    )
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rank_seed)
    return rank_seed


def random_orthogonal(batch_size, dim, device, dtype):
    q = torch.randn((batch_size, dim, dim), device=device, dtype=dtype)
    q, r = torch.linalg.qr(q)
    signs = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    signs[signs == 0] = 1
    d = torch.diag_embed(signs)
    return q @ d


@torch.no_grad()
def eval_means(model, dataset, metrics):
    model.eval()

    if hasattr(dataset, "means"):
        mus = dataset.means
        model_log_probs = -model(mus).reshape(-1).cpu().numpy()
        data_log_probs = dataset.log_prob(mus).reshape(-1).cpu().numpy()
        logging.info(f"Model log probs at means: {model_log_probs}")
        logging.info(f"Data log probs at means: {data_log_probs}")

        metrics["model_log_prob_at_means"] = model_log_probs
        metrics["data_log_prob_at_means"] = data_log_probs

    return metrics


# --- 2. The SciPy-PyTorch Bridge ---
# This class is the core of the implementation.
class SciPyOptimizer:
    """
    A wrapper that connects a PyTorch model to the SciPy optimizers.
    """

    def __init__(self, model, loss_fn, batch, device):
        """
        Initializes the wrapper.

        Args:
            model (nn.Module): The PyTorch model to optimize.
            loss_fn: The loss function.
            x_train (torch.Tensor): Training data features.
            y_train (torch.Tensor): Training data labels.
        """
        self.model = model
        self.loss_fn = loss_fn
        self.batch = batch
        # Store a reference to the model's parameters for easy access
        self.params = list(model.parameters())
        self.device = device

    def objective(self, x_np):
        """
        The objective function for SciPy's minimize.
        This function accepts a flat NumPy array of parameters, updates the model,
        computes the loss and gradients, and returns them.

        Args:
            x_np (np.ndarray): A flat NumPy array containing the model parameters.

        Returns:
            tuple: A tuple containing the scalar loss and a flat NumPy array
                   of gradients.
        """
        # 1. Convert NumPy array from SciPy to a PyTorch tensor
        params_torch = torch.from_numpy(x_np).to(self.device)

        # 2. Load the parameters into the model
        vector_to_parameters(params_torch, self.params)

        # 3. Perform a standard PyTorch forward and backward pass
        self.model.zero_grad()
        loss = self.loss_fn(self.model, self.batch)
        loss.backward()

        # 4. Flatten the gradients and convert them to a NumPy array
        grads_flat = parameters_to_vector([p.grad for p in self.params])

        # SciPy's L-BFGS-B requires float64
        return loss.item(), grads_flat.cpu().numpy().astype(np.float64)


def softplus_inverse(x):
    # https://github.com/pytorch/pytorch/issues/72759#issuecomment-1236496693
    return x + torch.log(-torch.expm1(-x))


class EBMFromTRERatio(nn.Module):
    def __init__(
        self,
        model,
        dim,
        num_noise_classes,
        final_flat=False,
        eval_dims=None,
    ):
        super(EBMFromTRERatio, self).__init__()
        self.model = model
        self.dim = dim
        if eval_dims is None:
            self.eval_dims = (dim,)
        else:
            self.eval_dims = eval_dims
        self.data_dims = [i + 1 for i in range(len(self.eval_dims))]
        self.num_noise_classes = int(num_noise_classes)
        self.final_flat = final_flat
        self.log_norm = 0.5 * self.dim * math.log(2.0 * math.pi)
        self.times = torch.arange(
            start=0,
            end=self.num_noise_classes,
            step=1,
            dtype=torch.long,
        )

    def base_energy(self, x):
        return self.log_norm + 0.5 * torch.sum(
            torch.square(x), dim=self.data_dims
        ).view(-1)

    def forward(self, x, t=None):
        current_base_energy = self.base_energy(x)

        output = self.model(x)

        if (t is None) or (t == 1.0):
            result = current_base_energy - torch.sum(output, dim=1, keepdim=False)
            if self.final_flat:
                return result
            else:
                return result.unsqueeze(dim=1)

        u = t * self.num_noise_classes
        k = math.floor(u)
        alpha = u - k
        energy_lower = current_base_energy - torch.sum(
            output[:, :k], dim=1, keepdim=False
        )
        energy_upper = current_base_energy - torch.sum(
            output[:, : k + 1], dim=1, keepdim=False
        )
        result = (1 - alpha) * energy_lower + alpha * energy_upper
        if self.final_flat:
            return result
        else:
            return result.unsqueeze(dim=1)


class EBMFromMultRatio(nn.Module):
    def __init__(
        self,
        model,
        dim,
        num_noise_classes,
        final_flat=False,
        eval_dims=None,
    ):
        super(EBMFromMultRatio, self).__init__()
        self.model = model
        self.dim = dim
        if eval_dims is None:
            self.eval_dims = (dim,)
        else:
            self.eval_dims = eval_dims
        self.data_dims = [i + 1 for i in range(len(self.eval_dims))]
        self.num_noise_classes = int(num_noise_classes)
        self.final_flat = final_flat
        self.log_norm = 0.5 * self.dim * math.log(2.0 * math.pi)
        self.times = torch.arange(
            start=0,
            end=self.num_noise_classes,
            step=1,
            dtype=torch.long,
        )

    def base_energy(self, x, eval_mode):
        return self.log_norm + 0.5 * torch.sum(
            torch.square(x), dim=self.data_dims
        ).view(-1)

    def forward(self, x, t=None):
        current_base_energy = self.base_energy(x)

        output = -self.model(x)

        if (t is None) or (t == 1.0):
            result = output[:, -1] - output[:, 0] + current_base_energy
            if self.final_flat:
                return result
            else:
                return result.unsqueeze(dim=1)

        raise NotImplementedError


class EBMFromNCEEnergy(nn.Module):
    def __init__(
        self,
        model,
        num_noise_classes,
        dim,
        data_shape,
        normalize_time=False,
        eps1=0.0,
        eps2=0.0,
    ):
        super(EBMFromNCEEnergy, self).__init__()
        self.model = model
        if (num_noise_classes == float("inf")) or normalize_time:
            self.default_noise_t = eps1
            self.default_t = 1.0 - eps2
        else:
            self.default_noise_t = 0
            self.default_t = num_noise_classes
        self.dim = dim
        self.data_dims = [i + 1 for i in range(len(data_shape))]
        self.log_norm = 0.5 * self.dim * math.log(2.0 * math.pi)
        self.normalize_time = normalize_time

    def forward(self, x, t=None):
        if t is None:
            t = self.default_t
        data_t = torch.full(
            (x.shape[0],),
            fill_value=t,
            device=x.device,
            dtype=torch.float,
        )
        noise_t = torch.full(
            (x.shape[0],), self.default_noise_t, device=x.device, dtype=torch.float
        )
        data_energy = self.model(x, data_t)
        noise_energy = self.model(x, noise_t)
        base_energy = self.log_norm + 0.5 * torch.sum(
            torch.square(x), dim=self.data_dims, keepdim=False
        ).unsqueeze(dim=-1)
        return data_energy - noise_energy + base_energy


class SimpleEBM(nn.Module):
    def __init__(self, model):
        super(SimpleEBM, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


class DummyEBM(nn.Module):
    def __init__(self, model, num_noise_classes, eps1=0.0, eps2=0.0):
        super(DummyEBM, self).__init__()
        self.model = model
        if num_noise_classes == float("inf"):
            self.default_t = 1.0 - eps2
        else:
            self.default_t = num_noise_classes

    def forward(self, x, t=None):
        if t is None:
            t = self.default_t
        data_t = torch.full(
            (x.shape[0],),
            fill_value=t,
            device=x.device,
            dtype=torch.float,
        )
        data_energy = self.model(x, data_t)
        return data_energy


class EBMFromSTNCEEnergy(nn.Module):
    def __init__(
        self,
        model,
        num_noise_classes,
        dim,
        data_shape,
        log_const=None,
        is_dataloader=None,
        accelerator=None,
        eps1=0.0,
        eps2=0.0,
    ):
        super(EBMFromSTNCEEnergy, self).__init__()
        self.model = model
        if num_noise_classes == float("inf"):
            self.default_noise_t = eps1
            self.default_t = 1.0 - eps2
        else:
            self.default_noise_t = 0
            self.default_t = num_noise_classes
        self.dim = dim
        self.data_shape = data_shape
        self.device = next(self.model.parameters()).device
        self.is_dataloader = is_dataloader
        self.accelerator = accelerator
        self.specified_log_const = self._coerce_log_const(log_const)
        self._estimated_log_constant = None

    def _coerce_log_const(self, log_const):
        if log_const is None:
            return None

        if callable(log_const):
            try:
                log_const = log_const(self.model)
            except TypeError:
                log_const = log_const()

        if log_const is None:
            return None
        if torch.is_tensor(log_const):
            return float(log_const.detach().item())
        return float(log_const)

    def _estimate_log_constant_from_proposal(self):

        proposal_dist = Independent(
            Normal(
                loc=torch.zeros((self.dim,), device=self.device),
                scale=torch.ones((self.dim,), device=self.device),
            ),
            reinterpreted_batch_ndims=1,
        )
        energy_fn = lambda x: self.model(
            x.view((x.shape[0], *self.data_shape)),
            torch.full(
                (x.shape[0],),
                fill_value=self.default_noise_t,
                device=x.device,
                dtype=torch.float,
            ),
        )

        if self.is_dataloader is not None:
            return estimate_log_const_importance_from_dataloader(
                energy_fn=energy_fn,
                proposal_log_prob_fn=lambda x: proposal_dist.log_prob(
                    x.view(x.shape[0], -1)
                ),
                is_dataloader=self.is_dataloader,
                device=self.device,
                accelerator=self.accelerator,
            )

        log_constant = estimate_log_const_importance(
            energy_fn=energy_fn,
            proposal_dist=proposal_dist,
            device=self.device,
            return_diagnostics=False,
        ).item()
        return log_constant

    def estimate_log_constant(self):
        if self.specified_log_const is not None:
            return self.specified_log_const

        if self._estimated_log_constant is None:
            self._estimated_log_constant = self._estimate_log_constant_from_proposal()
        return self._estimated_log_constant

    @property
    def has_specified_log_constant(self):
        return self.specified_log_const is not None

    def forward(self, x, t=None):
        if t is None:
            t = self.default_t
        data_t = torch.full(
            (x.shape[0],),
            fill_value=t,
            device=x.device,
            dtype=torch.float,
        )
        data_energy = self.model(x, data_t) + self.estimate_log_constant()
        return data_energy


def update_weighted_running_mean(
    running_mean, running_weight, batch_mean, batch_weight
):
    batch_mean = torch.as_tensor(
        batch_mean, device=running_mean.device, dtype=torch.float64
    )
    batch_weight = torch.as_tensor(
        batch_weight, device=running_weight.device, dtype=torch.float64
    )
    total_weight = running_weight + batch_weight
    if batch_weight.item() == 0:
        return running_mean, running_weight
    running_mean = running_mean + (batch_mean - running_mean) * (
        batch_weight / total_weight
    )
    return running_mean, total_weight


def merge_weighted_running_means(means, weights):
    merged_mean = torch.zeros((), device=means.device, dtype=torch.float64)
    merged_weight = torch.zeros((), device=weights.device, dtype=torch.float64)
    for mean, weight in zip(means.unbind(), weights.unbind()):
        merged_mean, merged_weight = update_weighted_running_mean(
            merged_mean, merged_weight, mean, weight
        )
    return merged_mean, merged_weight


def evaluate_likelihood(
    model,
    test_dataloader,
    scaler,
    gt_log_prob_fn,
    device,
    log_normalizer=0.0,
    dequantization=False,
    accelerator=None,
):
    avg_loss = torch.zeros((), device=device, dtype=torch.float64)
    total_samples = torch.zeros((), device=device, dtype=torch.float64)

    for batch in test_dataloader:
        batch = batch[0].to(device=device)

        # dequantization and scale
        if dequantization:
            batch = ((batch * 255.0) + torch.rand_like(batch)) / 256.0

        # gt_logp is defined for input roughly in the range of [0, 1]
        gt_logp = gt_log_prob_fn(batch).to(device=device)

        log_det = scaler_log_det(batch)
        batch = scaler(batch)

        neg_log_q = model(batch, t=None).squeeze(dim=-1) + log_normalizer - log_det

        diff = (gt_logp + neg_log_q).to(torch.float64)
        batch_loss = torch.mean(diff * diff)
        avg_loss, total_samples = update_weighted_running_mean(
            avg_loss, total_samples, batch_loss, batch.shape[0]
        )

    if accelerator is not None:
        gathered = accelerator.gather(
            torch.stack([avg_loss, total_samples]).unsqueeze(0)
        )
        avg_loss, total_samples = merge_weighted_running_means(
            gathered[:, 0], gathered[:, 1]
        )

    avg_loss_value = avg_loss.item()
    log_info(
        f"Average density estimation loss over test set: {avg_loss_value:.4f}",
        accelerator=accelerator,
    )

    return avg_loss_value


def evaluate_score(
    model,
    test_dataloader,
    scaler,
    gt_score_fn,
    device,
    prob_path,
    dequantization=False,
):
    avg_loss = torch.zeros((), device=device, dtype=torch.float64)
    total_samples = torch.zeros((), device=device, dtype=torch.float64)

    for batch in test_dataloader:
        batch = batch[0].to(device=device)

        # dequantization and scale
        if dequantization:
            batch = ((batch * 255.0) + torch.rand_like(batch)) / 256.0

        # gt_logp is defined for input roughly in the range of [0, 1]
        gt_score = gt_score_fn(batch).to(device=device)

        batch = scaler(batch)
        t = prob_path.sample_t(batch.shape[0], device=batch.device)

        xs = prob_path.sample(z=batch, t=t)

        vs = model(xs, t)
        model_score = prob_path.score_from_v(vs, xs, t)

        diff = (gt_score - model_score).to(torch.float64)
        batch_loss = torch.mean(diff * diff)
        avg_loss, total_samples = update_weighted_running_mean(
            avg_loss, total_samples, batch_loss, batch.shape[0]
        )

    avg_loss_value = avg_loss.item()
    log_info(f"Average density estimation loss over test set: {avg_loss_value:.4f}")

    return avg_loss_value


def evaluate_bpd(
    model,
    set_name,
    set_dataloader,
    scaler,
    device,
    log_normalizer=0.0,
    dequantization=False,
    accelerator=None,
    non_blocking=False,
):
    avg_bpd = torch.zeros((), device=device, dtype=torch.float64)
    total_samples = torch.zeros((), device=device, dtype=torch.float64)

    for batch in set_dataloader:
        batch = batch[0].to(device=device, non_blocking=non_blocking)

        # dequantization and scale
        if dequantization:
            batch = ((batch * 255.0) + torch.rand_like(batch)) / 256.0

        log_det = scaler_log_det(batch)
        batch = scaler(batch)

        neg_log_q = (
            model(batch, t=None).reshape((batch.shape[0],)) + log_normalizer - log_det
        )

        # 7.0 bacause we scaled from [0, 255] to [-1, 1]
        bpd = (
            neg_log_q.to(torch.float64).mean() / (math.log(2.0) * batch[0].numel())
            + 8.0
        )

        avg_bpd, total_samples = update_weighted_running_mean(
            avg_bpd, total_samples, bpd, batch.shape[0]
        )

    if accelerator is not None:
        gathered = accelerator.gather(
            torch.stack([avg_bpd, total_samples]).unsqueeze(0)
        )
        avg_bpd, total_samples = merge_weighted_running_means(
            gathered[:, 0], gathered[:, 1]
        )

    avg_bpd_value = avg_bpd.item()
    log_info(f"Average bpd over {set_name} set: {avg_bpd_value:.4f}", accelerator)

    return avg_bpd_value


def register_metric(metrics, step, metric_name, value):
    if metric_name not in metrics:
        metrics[metric_name] = dict()
    metrics[metric_name][step] = value
    return metrics


@torch.no_grad()
def estimate_image_log_constant_from_dataloader(
    model,
    proposal_dist,
    is_dataloader,
    device,
    accelerator=None,
):
    def energy_fn(batch_data):
        log_det = scaler_log_det(batch_data)
        scaled_batch_data = scaler(batch_data)
        neg_log_q = model(scaled_batch_data, t=None).reshape(batch_data.shape[0], -1)
        return neg_log_q.squeeze(dim=-1) - log_det

    return estimate_log_const_importance_from_dataloader(
        energy_fn=energy_fn,
        proposal_log_prob_fn=proposal_dist.log_prob,
        is_dataloader=is_dataloader,
        device=device,
        accelerator=accelerator,
    )


def evaluate_image_density_metrics(
    model,
    set_name,
    set_dataloader,
    data_dist,
    device,
    log_constant,
    dequantization=False,
    accelerator=None,
    non_blocking=False,
):
    avg_mse = torch.zeros((), device=device, dtype=torch.float64)
    avg_norm_mse = torch.zeros((), device=device, dtype=torch.float64)
    avg_norm_nll = torch.zeros((), device=device, dtype=torch.float64)
    avg_ratio = torch.zeros((), device=device, dtype=torch.float64)
    total_samples = torch.zeros((), device=device, dtype=torch.float64)
    for batch in set_dataloader:
        batch_data = batch[0].to(device=device, non_blocking=non_blocking)
        other_batch_data = batch[1].to(device=device, non_blocking=non_blocking)

        if dequantization:
            batch_data = ((batch_data * 255.0) + torch.rand_like(batch_data)) / 256.0
            other_batch_data = (
                (other_batch_data * 255.0) + torch.rand_like(other_batch_data)
            ) / 256.0

        gt_log_probs = data_dist.log_prob(batch_data).to(device=device).reshape(-1)
        log_det = scaler_log_det(batch_data)
        scaled_batch_data = scaler(batch_data)
        model_log_probs = (
            -model(scaled_batch_data, t=None).reshape((batch_data.shape[0],)) + log_det
        )

        normalized_model_log_probs = model_log_probs - log_constant
        mse_diff = (gt_log_probs - model_log_probs).to(torch.float64)
        batch_mse = torch.mean(mse_diff * mse_diff)
        norm_mse_diff = (gt_log_probs - normalized_model_log_probs).to(torch.float64)
        batch_norm_mse = torch.mean(norm_mse_diff * norm_mse_diff)
        batch_norm_nll = -torch.mean(normalized_model_log_probs.to(torch.float64))
        prev_total_samples = total_samples
        avg_mse, total_samples = update_weighted_running_mean(
            avg_mse, total_samples, batch_mse, batch_data.shape[0]
        )
        avg_norm_mse, _ = update_weighted_running_mean(
            avg_norm_mse, prev_total_samples, batch_norm_mse, batch_data.shape[0]
        )
        avg_norm_nll, _ = update_weighted_running_mean(
            avg_norm_nll, prev_total_samples, batch_norm_nll, batch_data.shape[0]
        )

        other_gt_log_probs = (
            data_dist.log_prob(other_batch_data).to(device=device).reshape(-1)
        )
        other_log_det = scaler_log_det(other_batch_data)
        other_scaled_batch_data = scaler(other_batch_data)
        other_neg_log_q = model(other_scaled_batch_data, t=None).reshape(
            other_batch_data.shape[0], -1
        )
        other_neg_log_q = other_neg_log_q.squeeze(dim=-1) - other_log_det
        other_model_log_probs = (-other_neg_log_q).reshape(-1)

        ratio_diff = (
            (other_gt_log_probs - gt_log_probs)
            - (other_model_log_probs - model_log_probs)
        ).to(torch.float64)
        batch_ratio = torch.mean(ratio_diff * ratio_diff)
        avg_ratio, _ = update_weighted_running_mean(
            avg_ratio, prev_total_samples, batch_ratio, batch_data.shape[0]
        )

    if accelerator is not None:
        gathered = accelerator.gather(
            torch.stack(
                [avg_mse, avg_norm_mse, avg_norm_nll, avg_ratio, total_samples]
            ).unsqueeze(0)
        )
        avg_mse, _ = merge_weighted_running_means(gathered[:, 0], gathered[:, 4])
        avg_norm_mse, _ = merge_weighted_running_means(gathered[:, 1], gathered[:, 4])
        avg_norm_nll, _ = merge_weighted_running_means(gathered[:, 2], gathered[:, 4])
        avg_ratio, total_samples = merge_weighted_running_means(
            gathered[:, 3], gathered[:, 4]
        )

    avg_mse_value = avg_mse.item()
    avg_norm_mse_value = avg_norm_mse.item()
    avg_norm_nll_value = avg_norm_nll.item()
    avg_ratio_value = avg_ratio.item()

    if set_name == "val":
        log_info(
            f"Average image MSE over {set_name} set: {avg_mse_value:.4f}", accelerator
        )
        log_info(
            f"Average normalized image MSE over {set_name} set: {avg_norm_mse_value:.4f}",
            accelerator,
        )
        log_info(
            f"Average normalized image NLL over {set_name} set: {avg_norm_nll_value:.4f}",
            accelerator,
        )
        log_info(
            f"Average image ratio loss over {set_name} set: {avg_ratio_value:.4f}",
            accelerator,
        )

    return avg_mse_value, avg_norm_mse_value, avg_norm_nll_value, avg_ratio_value


@torch.no_grad()
def image_evaluate(
    set_names,
    set_dataloaders,
    is_dataloader,
    device,
    use_dequantization,
    build_noise_fn,
    eval_loss_fn,
    data_dist,
    get_ebm,
    step,
    model,
    metrics,
    accelerator=None,
    non_blocking=False,
    return_log_densities=False,
):
    log_info(f"Step {step} evaluating:", accelerator)

    ebm_is_dataloader = is_dataloader if (data_dist is None) else None
    ebm_model = get_ebm(
        model,
        is_dataloader=ebm_is_dataloader,
        accelerator=accelerator,
    )
    current_log_constant = None
    if hasattr(ebm_model, "has_specified_log_constant") and (
        ebm_model.has_specified_log_constant
    ):
        current_log_constant = float(ebm_model.estimate_log_constant())
    elif (data_dist is None) and hasattr(ebm_model, "estimate_log_constant"):
        current_log_constant = float(ebm_model.estimate_log_constant())
    elif (data_dist is None) and hasattr(ebm_model, "log_constant"):
        current_log_constant = float(ebm_model.log_constant)

    shared_log_constant = None
    if data_dist is not None:
        if is_dataloader is None:
            raise ValueError(
                "Expected a dedicated importance-sampling dataloader when "
                "data_dist is available."
            )
        shared_log_constant = estimate_image_log_constant_from_dataloader(
            model=ebm_model,
            proposal_dist=data_dist,
            is_dataloader=is_dataloader,
            device=device,
            accelerator=accelerator,
        )
        if current_log_constant is not None:
            register_metric(metrics, step, "model_log_constant", current_log_constant)
        register_metric(metrics, step, "log_constant", shared_log_constant)
    elif current_log_constant is not None:
        register_metric(metrics, step, "log_constant", current_log_constant)

    log_density_outputs = {} if return_log_densities else None

    for set_name, set_dataloader in zip(set_names, set_dataloaders):
        avg_main_loss = torch.zeros((), device=device, dtype=torch.float64)
        avg_reg_loss = torch.zeros((), device=device, dtype=torch.float64)
        avg_loss = torch.zeros((), device=device, dtype=torch.float64)
        num_samples = torch.zeros((), device=device, dtype=torch.float64)
        set_log_density_parts = []
        set_log_density_index_parts = []
        for batch in set_dataloader:
            batch_data = batch[0].to(device=device, non_blocking=non_blocking)
            batch_size = batch_data.shape[0]

            # dequantization and scale
            if use_dequantization:
                batch_data = (
                    (batch_data * 255.0) + torch.rand_like(batch_data)
                ) / 256.0
            unscaled_batch_data = batch_data
            batch_data = scaler(unscaled_batch_data)

            if return_log_densities:
                sample_indices = batch[-1]
                if (not torch.is_tensor(sample_indices)) or (
                    sample_indices.shape[0] != batch_size
                ):
                    raise ValueError(
                        "return_log_densities=True expects the dataloader batch to "
                        "include sample indices as its final tensor."
                    )

                log_det = scaler_log_det(unscaled_batch_data)
                batch_log_densities = (
                    -ebm_model(batch_data, t=None).reshape((batch_size,)) + log_det
                ).to(torch.float64)
                if shared_log_constant is not None:
                    batch_log_densities = batch_log_densities - shared_log_constant

                sample_indices = sample_indices.to(device=device, dtype=torch.long)
                set_log_density_index_parts.append(sample_indices.detach().cpu())
                set_log_density_parts.append(batch_log_densities.detach().cpu())

            batch = build_noise_fn(
                original_data=batch_data, train=False, model_energy_fn=model
            )

            losses = eval_loss_fn(model, batch)
            if type(losses) == tuple:
                batch_main_loss, batch_reg_loss, batch_loss = losses
            else:
                batch_loss = losses
                batch_main_loss = batch_loss
                batch_reg_loss = torch.zeros_like(batch_loss)
            prev_num_samples = num_samples
            avg_main_loss, num_samples = update_weighted_running_mean(
                avg_main_loss, num_samples, batch_main_loss.detach(), batch_size
            )
            avg_reg_loss, _ = update_weighted_running_mean(
                avg_reg_loss, prev_num_samples, batch_reg_loss.detach(), batch_size
            )
            avg_loss, _ = update_weighted_running_mean(
                avg_loss, prev_num_samples, batch_loss.detach(), batch_size
            )

        if accelerator is not None:
            gathered = accelerator.gather(
                torch.stack(
                    [avg_main_loss, avg_reg_loss, avg_loss, num_samples]
                ).unsqueeze(0)
            )
            avg_main_loss, _ = merge_weighted_running_means(
                gathered[:, 0], gathered[:, 3]
            )
            avg_reg_loss, _ = merge_weighted_running_means(
                gathered[:, 1], gathered[:, 3]
            )
            avg_loss, _ = merge_weighted_running_means(gathered[:, 2], gathered[:, 3])

        register_metric(metrics, step, f"{set_name}_main_loss", avg_main_loss.item())
        register_metric(metrics, step, f"{set_name}_reg_loss", avg_reg_loss.item())
        register_metric(metrics, step, f"{set_name}_loss", avg_loss.item())

        if data_dist is not None:
            mse, norm_mse, norm_nll, ratio = evaluate_image_density_metrics(
                model=ebm_model,
                set_name=set_name,
                set_dataloader=set_dataloader,
                data_dist=data_dist,
                device=device,
                log_constant=shared_log_constant,
                dequantization=use_dequantization,
                accelerator=accelerator,
                non_blocking=non_blocking,
            )
            register_metric(metrics, step, f"{set_name}_mse", mse)
            register_metric(metrics, step, f"{set_name}_norm_mse", norm_mse)
            register_metric(metrics, step, f"{set_name}_norm_nll", norm_nll)
            register_metric(metrics, step, f"{set_name}_real_ratio", ratio)

        else:
            bpd = evaluate_bpd(
                model=ebm_model,
                set_name=set_name,
                set_dataloader=set_dataloader,
                scaler=scaler,
                device=device,
                dequantization=use_dequantization,
                accelerator=accelerator,
                non_blocking=non_blocking,
            )
            register_metric(metrics, step, f"{set_name}_bpd", bpd)

        if return_log_densities:
            if set_log_density_parts:
                sample_indices = torch.cat(set_log_density_index_parts)
                log_densities = torch.cat(set_log_density_parts)
            else:
                sample_indices = torch.empty((0,), dtype=torch.long)
                log_densities = torch.empty((0,), dtype=torch.float64)
            if accelerator is not None:
                local_count = torch.tensor(
                    [sample_indices.numel()], device=device, dtype=torch.long
                )
                all_counts = accelerator.gather(local_count)
                max_count = int(all_counts.max().item())

                padded_indices = torch.full(
                    (max_count,), -1, device=device, dtype=torch.long
                )
                padded_log_densities = torch.zeros(
                    (max_count,), device=device, dtype=torch.float64
                )
                if sample_indices.numel() > 0:
                    padded_indices[: sample_indices.numel()] = sample_indices.to(
                        device=device
                    )
                    padded_log_densities[: log_densities.numel()] = log_densities.to(
                        device=device
                    )

                gathered_indices = accelerator.gather(padded_indices)
                gathered_log_densities = accelerator.gather(padded_log_densities)
                valid_mask = gathered_indices >= 0
                sample_indices = gathered_indices[valid_mask].detach().cpu()
                log_densities = gathered_log_densities[valid_mask].detach().cpu()
            sort_order = torch.argsort(sample_indices, stable=True)
            sample_indices = sample_indices[sort_order]
            log_densities = log_densities[sort_order]
            if sample_indices.numel() > 1:
                keep = torch.ones(
                    sample_indices.shape[0], dtype=torch.bool, device="cpu"
                )
                keep[1:] = sample_indices[1:] != sample_indices[:-1]
                sample_indices = sample_indices[keep]
                log_densities = log_densities[keep]
            log_density_outputs[set_name] = {
                "indices": sample_indices,
                "log_densities": log_densities,
            }

    log_info("", accelerator)
    if return_log_densities:
        return metrics, log_density_outputs
    return metrics


@torch.no_grad()
def image_evaluate_fm(
    prob_path,
    gt_score_fn,
    set_names,
    set_dataloaders,
    step,
    model,
    build_noise_fn,
    eval_loss_fn,
    metrics,
    device,
    use_dequantization=False,
    eps=1e-4,
    accelerator=None,
):
    log_info(f"Step {step} evaluating:", accelerator)

    for set_name, set_dataloader in zip(set_names, set_dataloaders):
        loss = torch.zeros((), device=device, dtype=torch.float64)
        score_error = None
        if gt_score_fn is not None:
            score_error = torch.zeros((), device=device, dtype=torch.float64)
        num_samples = torch.zeros((), device=device, dtype=torch.float64)
        for batch in set_dataloader:
            batch_data = batch[0].to(device=device)
            batch_size = batch_data.shape[0]

            # dequantization and scale
            if use_dequantization:
                batch_data = (
                    (batch_data * 255.0) + torch.rand_like(batch_data)
                ) / 256.0
            batch_data = scaler(batch_data)

            batch = build_noise_fn(
                original_data=batch_data, train=False, model_energy_fn=model
            )

            # loss
            losses = eval_loss_fn(model, batch)
            if type(losses) == tuple:
                batch_loss = losses[-1].detach()
            else:
                batch_loss = losses.detach()
            prev_num_samples = num_samples
            loss, num_samples = update_weighted_running_mean(
                loss, num_samples, batch_loss, batch_size
            )

            if gt_score_fn is not None:
                # score error
                ts = prob_path.sample_t(batch_data.shape[0], device=batch_data.device)
                xs = prob_path.sample(z=batch_data, t=ts)
                model_scores = prob_path.score_from_v(vs=model(xs, ts), xs=xs, t=ts)
                gt_scores = gt_score_fn(xs, ts)

                # avoid blowing up
                batch_score_error = torch.mean(
                    torch.mean(
                        torch.square((model_scores - gt_scores).to(torch.float64)),
                        dim=[1, 2, 3],
                    )
                    * torch.square(prob_path.beta(ts).to(torch.float64))
                )
                score_error, _ = update_weighted_running_mean(
                    score_error, prev_num_samples, batch_score_error, batch_size
                )

        if accelerator is not None:
            if gt_score_fn is not None:
                gathered = accelerator.gather(
                    torch.stack([loss, score_error, num_samples]).unsqueeze(0)
                )
                loss, _ = merge_weighted_running_means(gathered[:, 0], gathered[:, 2])
                score_error, _ = merge_weighted_running_means(
                    gathered[:, 1], gathered[:, 2]
                )
            else:
                gathered = accelerator.gather(
                    torch.stack([loss, num_samples]).unsqueeze(0)
                )
                loss, _ = merge_weighted_running_means(gathered[:, 0], gathered[:, 1])

        metric_name = f"{set_name}_loss"
        if set_name == "val":
            log_info(f"{metric_name}: {loss.item():.3f}", accelerator)
        register_metric(metrics, step, metric_name, loss.item())

        if gt_score_fn is not None:
            metric_name = f"{set_name}_score"
            if set_name == "val":
                log_info(f"{metric_name}: {score_error.item():.3f}", accelerator)
            register_metric(metrics, step, metric_name, score_error.item())

    log_info("", accelerator)
    return metrics


class VelocityWrapper(ModelWrapper):
    def __init__(self, model, prob_path, gt_scores=False):
        super().__init__(model)
        self.prob_path = prob_path
        self.gt_scores = gt_scores
        if self.gt_scores:
            # gt scores
            data_statistics = self.prob_path.data_statistics
            self.data_means = data_statistics["means"]
            self.flat_data_means = self.data_means.view((self.data_means.shape[0], -1))
            self.data_std = data_statistics["std"]

    @staticmethod
    def _batch_time(t, x):
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=x.device, dtype=x.dtype)
        else:
            t = t.to(device=x.device)

        if t.ndim == 0:
            return t.expand(x.shape[0])

        t = t.reshape(-1)
        if t.shape[0] == 1 and x.shape[0] != 1:
            return t.expand(x.shape[0])
        if t.shape[0] != x.shape[0]:
            raise ValueError(
                f"Time batch has shape {tuple(t.shape)}, but input batch has "
                f"shape {tuple(x.shape)}."
            )
        return t

    def forward(self, x, t):
        t = self._batch_time(t, x)
        if self.gt_scores:
            if self.data_means.ndim == 1:
                scores = gaussian_score(
                    x.view((-1, self.prob_path.dim)),
                    t=t,
                    mus=self.flat_data_means,
                    sigma=self.data_std,
                    alpha_fn=self.prob_path.alpha,
                    beta_fn=self.prob_path.beta,
                ).view(x.shape)
            else:
                scores = gaussian_mixture_score(
                    x.view((-1, self.prob_path.dim)),
                    t=t,
                    mus=self.flat_data_means,
                    sigma=self.data_std,
                    alpha_fn=self.prob_path.alpha,
                    beta_fn=self.prob_path.beta,
                ).view(x.shape)
            return self.prob_path.v_from_score(scores, x, t)
        else:
            with torch.enable_grad():
                x.requires_grad_(True)
                energies = self.model(x, t)
                scores = -torch.autograd.grad(energies.sum(), x, create_graph=False)[
                    0
                ].detach()
            x = x.detach()
            return self.prob_path.v_from_score(scores, x, t)


def euler_sample(
    velocity_wrapper, num_samples, batch_size, data_shape, num_steps, device
):
    all_samples = []
    ode_solver = ODESolver(velocity_wrapper)
    for i in range(math.ceil(num_samples / batch_size)):
        current_batch_size = min(batch_size, num_samples - i * batch_size)
        x_init = torch.randn((current_batch_size, *data_shape), device=device)
        samples = ode_solver.sample(x_init=x_init, step_size=1.0 / num_steps)
        all_samples.append(samples)
    return torch.cat(all_samples, dim=0)
