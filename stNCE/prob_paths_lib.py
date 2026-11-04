import math
import torch
from typing import Callable, Dict
from analytics import (
    gaussian_get_posterior_params,
    gmm_get_posterior_params,
    gaussian_sample,
    gaussian_log_prob,
    gaussian_score,
    gaussian_time_score,
    gmm_sample,
    gmm_log_prob,
    gaussian_mixture_score,
    gaussian_mixture_time_score,
)

sqrt_two = math.sqrt(2.0)


class FixedNormal:
    def __init__(self, dim, std, reduce_dims):
        self.dim = dim
        self.log_norm = -0.5 * self.dim * math.log(2.0 * math.pi)
        self.std = std
        self.log_std = math.log(std)
        self.reduce_dims = reduce_dims

    def log_prob(self, x, means):
        return (
            self.log_norm
            - self.dim * self.log_std
            - 0.5
            * torch.sum(torch.square((x - means) / self.std), dim=self.reduce_dims)
        )


class FlexNormal:
    def __init__(self, dim, reduce_dims):
        self.dim = dim
        self.log_norm = -0.5 * self.dim * math.log(2.0 * math.pi)
        self.reduce_dims = reduce_dims

    def log_prob(self, x, means, stds):
        return (
            self.log_norm
            - self.dim * torch.log(stds.squeeze(dim=self.reduce_dims))
            - 0.5 * torch.sum(torch.square((x - means) / stds), dim=self.reduce_dims)
        )


class VP:
    def __init__(self, num_steps, is_image, dim, device, stnce_noise_type="scale"):
        self.num_steps = num_steps
        eps = 1e-3
        self.betas = torch.linspace(1.0 - eps, eps, num_steps, device=device)
        self.alphas = torch.sqrt(1.0 - torch.square(self.betas))
        if is_image:
            self.dims = (1, 1, 1)
        else:
            self.dims = (1,)
        self.normal = FixedNormal(
            dim=dim, std=1.0, reduce_dims=tuple(range(-len(self.dims), 0))
        )  # TODO

    def _expand_batch(self, values):
        """Broadcast batch-wise scalars across the data dimensions."""
        return values.view((-1, *self.dims))

    def sample(self, z, t):
        # z: clean data
        # t: time step, integer between 0 and num_steps-1, t=0 means noise
        alphas = self._expand_batch(self.alphas[t])
        betas = self._expand_batch(self.betas[t])
        return alphas * z + betas * torch.randn_like(z)

    def conditional_sample(self, x, t, tau):
        new_alphas = self.alphas[tau] / self.alphas[t]
        new_betas = torch.sqrt(
            torch.abs(
                torch.square(self.betas[tau])
                - torch.square(new_alphas) * torch.square(self.betas[t])
            )
        )
        new_alphas = new_alphas.unsqueeze(dim=-1)
        new_betas = new_betas.unsqueeze(dim=-1)
        noise = torch.randn_like(x)
        return new_alphas * x + new_betas * noise

    def conditional_log_prob(self, x, t, y, tau):
        new_alphas = self.alphas[tau] / self.alphas[t]
        new_betas = torch.sqrt(
            torch.abs(
                torch.square(self.betas[tau])
                - torch.square(new_alphas) * torch.square(self.betas[t])
            )
        )
        new_alphas = new_alphas.unsqueeze(dim=-1)
        new_betas = new_betas.unsqueeze(dim=-1)
        return self.normal.log_prob(y, new_alphas * x, new_betas)

    def grouped_sample(self, x, t):
        if t.ndim != 1:
            raise ValueError(f"t must be 1D (K,), got shape {tuple(t.shape)}")

        # Share the same noise for a given datapoint across all time points.
        noise = torch.randn(
            (x.shape[0], t.shape[0], *x.shape[1:]), device=x.device, dtype=x.dtype
        )

        K = t.shape[0]
        broadcast_shape = (1, K) + self.dims

        alphas = self.alphas[t].view(broadcast_shape)
        betas = self.betas[t].view(broadcast_shape)

        return alphas * x.unsqueeze(dim=1) + betas * noise


def _transition_stats(prob_path, t, tau, eps):
    new_alphas = prob_path.alpha(tau) / prob_path.alpha(
        t
    )  # t and tau are guarded from 0 already

    new_betas = torch.clamp(
        torch.sqrt(
            torch.abs(
                torch.square(prob_path.beta(tau))
                - torch.square(new_alphas) * torch.square(prob_path.beta(t))
            )
        ),
        min=eps,
    )  # avoid numerical issues

    return prob_path._expand_batch(new_alphas), prob_path._expand_batch(new_betas)


def _scale(prob_path, eps, std=None):
    del std

    @torch.no_grad()
    def conditional_sample(x, t, tau, scores=None):
        del scores
        new_alphas, new_betas = _transition_stats(prob_path, t, tau, eps)
        noise = torch.randn_like(x)
        return new_alphas * x + new_betas * noise

    @torch.no_grad()
    def conditional_log_prob(x, t, y, tau, scores=None):
        del scores
        new_alphas, new_betas = _transition_stats(prob_path, t, tau, eps)
        return prob_path.normal.log_prob(y, new_alphas * x, new_betas)

    return conditional_sample, conditional_log_prob


def _white(prob_path, eps, std):
    del prob_path, eps

    @torch.no_grad()
    def conditional_sample(x, t, tau, scores=None):
        del t, tau, scores
        return x + std * torch.randn_like(x)

    @torch.no_grad()
    def conditional_log_prob(x, t, y, tau, scores=None):
        del x, y, tau, scores
        return torch.zeros_like(t)

    return conditional_sample, conditional_log_prob


def _oracle(prob_path, eps, std=None):
    del std
    data_means = prob_path.data_statistics["means"]
    flat_data_means = data_means.view((data_means.shape[0], -1))
    data_std = prob_path.data_statistics["std"]

    if data_means.ndim == 1:
        get_posterior_params = gaussian_get_posterior_params
        get_sample = gaussian_sample
        get_log_prob = gaussian_log_prob
    else:
        get_posterior_params = gmm_get_posterior_params
        get_sample = gmm_sample
        get_log_prob = gmm_log_prob

    @torch.no_grad()
    def conditional_sample(x, t, tau, scores=None):
        del scores

        result = torch.zeros_like(x)
        mask_greater = tau > t

        if torch.any(~mask_greater):
            x_sub = x[~mask_greater].view(-1, prob_path.dim)
            t_sub = t[~mask_greater].view(-1)
            tau_sub = tau[~mask_greater].view(-1)
            new_alphas, new_betas = _transition_stats(prob_path, t_sub, tau_sub, eps)
            noise = torch.randn_like(x_sub)
            result[~mask_greater] = new_alphas * x_sub + new_betas * noise

        if torch.any(mask_greater):
            x_sub = x[mask_greater].view(-1, prob_path.dim)
            t_sub = t[mask_greater].view(-1)
            tau_sub = tau[mask_greater].view(-1)
            variables = get_posterior_params(
                means=flat_data_means,
                std=data_std,
                alpha_fn=prob_path.alpha,
                beta_fn=prob_path.beta,
                x=x_sub,
                t=t_sub,
                tau=tau_sub,
            )
            result[mask_greater] = get_sample(*variables)

        return result

    @torch.no_grad()
    def conditional_log_prob(x, t, y, tau, scores=None):
        del scores

        result = torch.zeros_like(t, device=x.device)
        mask_greater = tau > t

        if torch.any(~mask_greater):
            x_sub = x[~mask_greater].view(-1, prob_path.dim)
            t_sub = t[~mask_greater].view(-1)
            y_sub = y[~mask_greater].view(-1, prob_path.dim)
            tau_sub = tau[~mask_greater].view(-1)
            new_alphas, new_betas = _transition_stats(prob_path, t_sub, tau_sub, eps)

            result[~mask_greater] = prob_path.normal.log_prob(
                y_sub, new_alphas * x_sub, new_betas
            )

        if torch.any(mask_greater):
            x_sub = x[mask_greater].view(-1, prob_path.dim)
            t_sub = t[mask_greater].view(-1)
            y_sub = y[mask_greater].view(-1, prob_path.dim)
            tau_sub = tau[mask_greater].view(-1)
            variables = get_posterior_params(
                means=flat_data_means,
                std=data_std,
                alpha_fn=prob_path.alpha,
                beta_fn=prob_path.beta,
                x=x_sub,
                t=t_sub,
                tau=tau_sub,
            )
            result[mask_greater] = get_log_prob(*variables, y_sub)

        return result

    return conditional_sample, conditional_log_prob


def _recovery_diff(prob_path, eps, std=None):
    del std

    @torch.no_grad()
    def conditional_sample(x, t, tau, scores=None):

        mask_greater = tau > t

        new_alphas, new_betas = _transition_stats(prob_path, t, tau, eps)
        noise = torch.randn_like(x)
        correction = torch.zeros_like(x)
        if scores is None and torch.any(mask_greater):
            scores = prob_path.get_target_score(x=x[mask_greater], t=t[mask_greater])
        if scores is not None:
            correction[mask_greater] = torch.square(new_betas[mask_greater]) * scores
        return new_alphas * x + correction + new_betas * noise

    @torch.no_grad()
    def conditional_log_prob(x, t, y, tau, scores=None):

        mask_greater = tau > t

        new_alphas, new_betas = _transition_stats(prob_path, t, tau, eps)

        correction = torch.zeros_like(x)
        if (scores is None) and torch.any(mask_greater):
            scores = prob_path.get_target_score(x=x[mask_greater], t=t[mask_greater])
        if scores is not None:
            correction[mask_greater] = torch.square(new_betas[mask_greater]) * scores

        return prob_path.normal.log_prob(y, new_alphas * x + correction, new_betas)

    return conditional_sample, conditional_log_prob


def _transition_stats_ei_sde(prob_path, t, tau, eps, mode="original"):
    if mode == "original":
        # 1. Linear scaling factor (alpha_tau / alpha_t)
        new_alphas = prob_path.alpha(tau) / prob_path.alpha(
            t
        )  # t and tau are guarded from 0 already

        # Fetch raw schedule standard deviations
        beta_t = prob_path.beta(t)
        beta_tau = prob_path.beta(tau)

        # 2. Stochastic noise standard deviation (matches exact Bayes variance)
        new_betas = torch.clamp(
            torch.sqrt(
                torch.abs(
                    torch.square(beta_tau)
                    - torch.square(new_alphas) * torch.square(beta_t)
                )
            ),
            min=eps,
        )  # avoid numerical issues

        # 3. Exponential Integrator score coefficient
        ei_score_coeff = 2.0 * beta_t * (new_alphas * beta_t - beta_tau)

    elif mode == "condot":
        new_alphas = tau / t  # t and tau are guarded from 0 already
        beta_t = 1.0 - t
        beta_tau = 1.0 - tau
        scaled_beta_t = new_alphas * beta_t

        # CondOT has alpha(t) = t and beta(t) = 1 - t, so
        # (alpha_tau / alpha_t) * beta_t - beta_tau = (tau - t) / t.
        beta_delta = (tau - t) / t
        transition_var = torch.abs(beta_delta) * (scaled_beta_t + beta_tau)
        new_betas = torch.clamp(
            torch.sqrt(torch.clamp_min(transition_var, 0.0)), min=eps
        )
        ei_score_coeff = 2.0 * beta_t * beta_delta
    else:
        raise ValueError(
            "ei_sde_transition_stats must be either 'original' or 'condot', "
            f"got {mode!r}."
        )

    return (
        prob_path._expand_batch(new_alphas),
        prob_path._expand_batch(new_betas),
        prob_path._expand_batch(ei_score_coeff),
    )


def _diff(prob_path, eps, std=None):
    del std

    @torch.no_grad()
    def conditional_sample(x, t, tau, scores=None):
        mask_greater = tau > t

        # Unpack all three quantities directly from the new transition function
        new_alphas, new_betas, ei_score_coeff = _transition_stats_ei_sde(
            prob_path, t, tau, eps, mode=prob_path.ei_sde_transition_stats
        )

        noise = torch.randn_like(x)
        correction = torch.zeros_like(x)

        if scores is None and torch.any(mask_greater):
            scores = prob_path.get_target_score(x=x[mask_greater], t=t[mask_greater])

        if scores is not None:
            # Apply the pre-computed EI coefficient to the scores
            correction[mask_greater] = ei_score_coeff[mask_greater] * scores

        return new_alphas * x + correction + new_betas * noise

    @torch.no_grad()
    def conditional_log_prob(x, t, y, tau, scores=None):
        mask_greater = tau > t

        # Unpack all three quantities
        new_alphas, new_betas, ei_score_coeff = _transition_stats_ei_sde(
            prob_path, t, tau, eps, mode=prob_path.ei_sde_transition_stats
        )

        correction = torch.zeros_like(x)

        if (scores is None) and torch.any(mask_greater):
            scores = prob_path.get_target_score(x=x[mask_greater], t=t[mask_greater])

        if scores is not None:
            # Apply the pre-computed EI coefficient for the likelihood mean
            correction[mask_greater] = ei_score_coeff[mask_greater] * scores

        return prob_path.normal.log_prob(y, new_alphas * x + correction, new_betas)

    return conditional_sample, conditional_log_prob


def _inf_diff(prob_path, eps, std=None):
    del eps, std

    @torch.no_grad()
    def conditional_sample(x, t, tau, scores=None):
        diff_t = prob_path.to_diffusion_time(t)
        diff_tau = prob_path.to_diffusion_time(tau)
        reverse_mask = diff_tau < diff_t
        forward_mask = ~reverse_mask

        t = prob_path._expand_batch(t)
        diff_t = prob_path._expand_batch(diff_t)
        diff_tau = prob_path._expand_batch(diff_tau)

        delta_t = torch.abs(diff_tau - diff_t)
        std_dt = torch.sqrt(delta_t)

        fs = prob_path.f(diff_t)
        gs, sq_gs = prob_path.g(diff_t)

        if scores is None and torch.any(reverse_mask):
            scores = prob_path.get_target_score(
                x=x[reverse_mask], t=t[reverse_mask].view(-1)
            )

        noise = torch.randn_like(x)
        samples = x + gs * std_dt * noise

        # reverse
        if torch.any(reverse_mask):
            samples[reverse_mask] = (
                samples[reverse_mask]
                - (fs[reverse_mask] * x[reverse_mask] - sq_gs[reverse_mask] * scores)
                * delta_t[reverse_mask]
            )
        # forward
        if torch.any(forward_mask):
            samples[forward_mask] = (
                samples[forward_mask]
                + fs[forward_mask] * x[forward_mask] * delta_t[forward_mask]
            )

        return samples

    @torch.no_grad()
    def conditional_log_prob(x, t, y, tau, scores=None):
        diff_t = prob_path.to_diffusion_time(t)
        diff_tau = prob_path.to_diffusion_time(tau)
        reverse_mask = diff_tau < diff_t
        forward_mask = ~reverse_mask

        t = prob_path._expand_batch(t)
        diff_t = prob_path._expand_batch(diff_t)
        diff_tau = prob_path._expand_batch(diff_tau)

        delta_t = torch.abs(diff_tau - diff_t)
        std_dt = torch.sqrt(delta_t)

        fs = prob_path.f(diff_t)
        gs, sq_gs = prob_path.g(diff_t)

        if scores is None and torch.any(reverse_mask):
            scores = prob_path.get_target_score(
                x=x[reverse_mask], t=t[reverse_mask].view(-1)
            )

        stds = gs * std_dt
        means = x.clone()
        # reverse
        if torch.any(reverse_mask):
            means[reverse_mask] = (
                means[reverse_mask]
                - (fs[reverse_mask] * x[reverse_mask] - sq_gs[reverse_mask] * scores)
                * delta_t[reverse_mask]
            )
        # forward
        if torch.any(forward_mask):
            means[forward_mask] = (
                means[forward_mask]
                + fs[forward_mask] * x[forward_mask] * delta_t[forward_mask]
            )

        return prob_path.normal.log_prob(y, means, stds)

    return conditional_sample, conditional_log_prob


_cont_kernel_registry: Dict[str, Callable] = {
    "scale": _scale,
    "white": _white,
    "oracle": _oracle,
    "diff": _diff,
    "recovery-diff": _recovery_diff,
    "inf-diff": _inf_diff,
}


class ProbPath:
    def __init__(
        self,
        is_image,
        dim,
        device,
        stnce_noise_type="scale",
        score_type="oracle",
        time_score_type="oracle",
        data_statistics=None,
        fm_model=None,
        eps=1e-4,
        ctsm_c=1.0,
        ei_sde_transition_stats="original",
    ):
        self.dim = dim
        self.device = device
        if stnce_noise_type.startswith("white"):
            stnce_noise_type, std = stnce_noise_type.split("-")
            self.std = float(std)
        else:
            self.std = None  # place holder
        self.eps = eps
        if ei_sde_transition_stats not in ("original", "condot"):
            raise ValueError(
                "ei_sde_transition_stats must be either 'original' or 'condot', "
                f"got {ei_sde_transition_stats!r}."
            )
        self.ei_sde_transition_stats = ei_sde_transition_stats
        self.stnce_noise_type = stnce_noise_type
        self.score_type = score_type
        self.time_score_type = time_score_type
        self.data_statistics = data_statistics
        builder = _cont_kernel_registry[self.stnce_noise_type]
        if is_image:
            self.dims = (1, 1, 1)
        else:
            self.dims = (1,)
        self.reduce_dims = tuple(range(-len(self.dims), 0))
        self.normal = FlexNormal(dim=dim, reduce_dims=self.reduce_dims)
        self.fm_model = fm_model
        self._cond_sample, self._cond_log_prob = builder(
            self, eps=self.eps, std=self.std
        )
        self.ctsm_c = ctsm_c

        self.data_means = self.data_statistics["means"]
        self.data_std = self.data_statistics["std"]
        self.flat_data_means = self.data_means.view((self.data_means.shape[0], -1))

    def _expand_batch(self, values):
        """Broadcast batch-wise scalars across the data dimensions."""
        return values.view((-1, *self.dims))

    def alpha(self, t):
        raise NotImplementedError

    def d_alpha(self, t):
        raise NotImplementedError

    def beta(self, t):
        raise NotImplementedError

    def d_beta(self, t):
        raise NotImplementedError

    def beta_var_d(self, t):
        raise NotImplementedError

    def f(self, t):
        raise NotImplementedError

    def g(self, t):
        raise NotImplementedError

    def to_diffusion_time(self, t):
        return 1.0 - t

    def sample(self, z, t):
        return self._expand_batch(self.alpha(t)) * z + self._expand_batch(
            self.beta(t)
        ) * torch.randn_like(z)

    def sample_and_noises(self, z, t):
        noises = torch.randn_like(z)
        return (
            self._expand_batch(self.alpha(t)) * z
            + self._expand_batch(self.beta(t)) * noises,
            noises,
        )

    def sample2(self, z, t1, t2):
        noises = torch.randn_like(z)
        return (
            self._expand_batch(self.alpha(t1)) * z
            + self._expand_batch(self.beta(t1)) * noises,
            self._expand_batch(self.alpha(t2)) * z
            + self._expand_batch(self.beta(t2)) * noises,
        )

    def sample2_and_noises(self, z, t1, t2):
        noises = torch.randn_like(z)
        return (
            self._expand_batch(self.alpha(t1)) * z
            + self._expand_batch(self.beta(t1)) * noises,
            self._expand_batch(self.alpha(t2)) * z
            + self._expand_batch(self.beta(t2)) * noises,
            noises,
        )

    def conditional_sample(self, x, t, tau, scores=None):
        return self._cond_sample(x, t, tau, scores)

    def conditional_log_prob(self, x, t, y, tau, scores=None):
        return self._cond_log_prob(x, t, y, tau, scores)

    def recover_noises(self, x, z, t):
        alphas = self._expand_batch(self.alpha(t))
        betas = self._expand_batch(self.beta(t))
        return (x - alphas * z) / betas

    def sample_and_stein_score(self, z, t, temp=None):
        # z: clean data
        # t: time step, float between 0 and 1
        # return noisy data, score and score weights
        # regress against score of energy
        alphas = self._expand_batch(self.alpha(t))
        betas = self._expand_batch(self.beta(t))
        if temp is not None:
            xs, noises = temp
        else:
            noises = torch.randn_like(z)
            xs = alphas * z + betas * noises
        targets = noises
        weights = betas
        return xs, targets, weights

    def handle_time_score_singularities(self, t, targets, weights, noises_z, noises_2):
        raise NotImplementedError

    def sample_and_time_score(self, z, t, temp=None):
        # z: clean data
        # t: time step, float between 0 and 1
        # return noisy data, score and score weights
        alphas = self.alpha(t)
        betas, vars, d_vars = self.beta_var_d(t)
        if temp is not None:
            xs, noises = temp
        else:
            noises = torch.randn_like(z)
            xs = self._expand_batch(alphas) * z + self._expand_batch(betas) * noises

        d_alphas = self.d_alpha(t)

        temp = self.dim * torch.sqrt(
            torch.square(d_vars) + 2 * vars * torch.square(d_alphas) * self.ctsm_c
        )
        noises_z = torch.sum(noises * z, dim=self.dims, keepdim=False)
        noises_2 = torch.sum(torch.square(noises), dim=self.dims, keepdim=False)
        targets = (
            -(
                -self.dim * d_vars / sqrt_two
                + sqrt_two * betas * d_alphas * noises_z
                + d_vars / sqrt_two * noises_2
            )
            / temp
        )

        weights = sqrt_two * vars / temp

        targets, weights = self.handle_time_score_singularities(
            t, targets, weights, noises_z, noises_2
        )
        return xs, targets, weights

    def sample_and_velocities(self, z, t):
        x0 = torch.randn_like(z)
        xs = (
            self._expand_batch(self.alpha(t)) * z
            + self._expand_batch(self.beta(t)) * x0
        )
        vs = (
            self._expand_batch(self.d_alpha(t)) * z
            + self._expand_batch(self.d_beta(t)) * x0
        )
        return xs, vs

    # @torch.no_grad()
    def ground_truth_score(self, x, t):
        if self.data_statistics is None:
            raise ValueError("Ground-truth scores require data_statistics.")
        if x.shape[0] == 0:
            return torch.empty_like(x)

        flat_x = x.view((x.shape[0], -1))
        flat_t = t.view(-1)

        if self.data_means.ndim == 1:
            scores = gaussian_score(
                x=flat_x,
                t=flat_t,
                mus=self.data_means,
                sigma=self.data_std,
                alpha_fn=self.alpha,
                beta_fn=self.beta,
            )
        else:
            scores = gaussian_mixture_score(
                x=flat_x,
                t=flat_t,
                mus=self.flat_data_means,
                sigma=self.data_std,
                alpha_fn=self.alpha,
                beta_fn=self.beta,
            )

        return scores.view(x.shape)

    def gaussian_score(self, x, t):
        if x.shape[0] == 0:
            return torch.empty_like(x)

        flat_x = x.view((x.shape[0], -1))
        flat_t = t.view(-1)

        scores = gaussian_score(
            x=flat_x,
            t=flat_t,
            mus=torch.zeros_like(flat_x[0]),
            sigma=1.0,
            alpha_fn=self.alpha,
            beta_fn=self.beta,
        )

        return scores.view(x.shape)

    # @torch.no_grad()
    def get_target_score(self, x, t):
        if self.score_type == "self":
            raise NotImplementedError(
                "get_target_score is not supported for self noise type."
            )
        elif self.score_type == "model":
            return self.score_from_v(self.fm_model(x, t), x, t)
        elif self.score_type == "oracle":
            return self.ground_truth_score(x, t)
        elif self.score_type == "flat":
            return torch.zeros_like(x)
        elif self.score_type == "gaussian":
            return self.gaussian_score(x, t)
        else:
            raise NotImplementedError

    def score_from_v(self, vs, xs, t, eps=1e-4):
        t = torch.clamp(t, 0.0, 1.0 - eps)
        alphas = self._expand_batch(self.alpha(t))
        betas = self._expand_batch(self.beta(t))
        d_alphas = self._expand_batch(self.d_alpha(t))
        d_betas = self._expand_batch(self.d_beta(t))

        scores = (alphas * vs - d_alphas * xs) / (
            d_alphas * torch.square(betas) - d_betas * betas * alphas
        )
        return scores

    def v_from_score(self, scores, xs, t, eps=1e-4):
        raise NotImplementedError

    # @torch.no_grad()
    def get_target_velocity(self, x, t):
        if self.score_type == "self":
            raise NotImplementedError(
                "get_target_score is not supported for self noise type."
            )
        elif self.score_type == "model":
            return self.fm_model(x, t)
        elif self.score_type == "oracle":
            return self.v_from_score(self.ground_truth_score(x, t), x, t)
        else:
            raise NotImplementedError

    # @torch.no_grad()
    def ground_truth_time_score(self, x, t):
        if self.data_statistics is None:
            raise ValueError("Ground-truth scores require data_statistics.")
        if x.shape[0] == 0:
            return torch.empty_like(x)

        data_means = self.data_statistics["means"]
        data_std = self.data_statistics["std"]
        flat_x = x.view((x.shape[0], -1))
        flat_data_means = data_means.view((data_means.shape[0], -1))
        flat_t = t.view(-1)

        if data_means.ndim == 1:
            scores = gaussian_time_score(
                x=flat_x,
                t=flat_t,
                mus=data_means,
                sigma=data_std,
                alpha_fn=self.alpha,
                beta_fn=self.beta,
                d_alpha_fn=self.d_alpha,
                d_beta_fn=self.d_beta,
            )
        else:
            scores = gaussian_mixture_time_score(
                x=flat_x,
                t=flat_t,
                mus=flat_data_means,
                sigma=data_std,
                alpha_fn=self.alpha,
                beta_fn=self.beta,
                d_alpha_fn=self.d_alpha,
                d_beta_fn=self.d_beta,
            )

        return scores.view_as(t)

    def get_target_time_score(self, x, t):
        if self.time_score_type == "oracle":
            return self.ground_truth_time_score(x, t)
        elif self.time_score_type == "fp":
            return self.fp_time_score(x, t)
        else:
            raise NotImplementedError

    def fp_time_score(self, x, t):
        with torch.no_grad():
            score = self.get_target_score(x, t)

        with torch.enable_grad():
            x.requires_grad_(True)
            velocity = self.get_target_velocity(x, t)

            # Hutchinson trace estimator with Rademacher
            # based on https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/solver/ode_solver.py
            z = (torch.randn_like(x).to(x.device) < 0) * 2.0 - 1.0
            ut_dot_z = torch.einsum(
                "ij,ij->i", velocity.flatten(start_dim=1), z.flatten(start_dim=1)
            )
            output = ut_dot_z.sum()
            grad_ut_dot_z = torch.autograd.grad(
                output,
                x,
                grad_outputs=torch.ones_like(output).detach(),
                create_graph=False,
                retain_graph=False,
            )[0]
            div = torch.einsum(
                "ij,ij->i", grad_ut_dot_z.flatten(start_dim=1), z.flatten(start_dim=1)
            )
        return (-div - torch.sum(velocity * score, dim=self.reduce_dims)).detach()


class CondOT(ProbPath):
    def __init__(
        self,
        is_image,
        dim,
        device,
        stnce_noise_type="scale",
        score_type="oracle",
        time_score_type="oracle",
        data_statistics=None,
        fm_model=None,
        eps1=0.0,
        eps2=0.0,
        eps=1e-4,
        t_dist="uniform",
        ctsm_c=1.0,
        rne=False,
        time_std=0.0,
        ei_sde_transition_stats="original",
    ):
        super().__init__(
            is_image=is_image,
            dim=dim,
            device=device,
            stnce_noise_type=stnce_noise_type,
            score_type=score_type,
            time_score_type=time_score_type,
            data_statistics=data_statistics,
            fm_model=fm_model,
            eps=eps,
            ctsm_c=ctsm_c,
            ei_sde_transition_stats=ei_sde_transition_stats,
        )
        self.noise_t = 0.0
        self.data_t = 1.0
        self.eps_factor = 1.0 - eps1 - eps2
        self.eps1 = eps1
        self.diff_eps2 = 1.0 - eps2
        self.t_dist = t_dist
        self.ctsm_num = math.sqrt(2.0 + self.ctsm_c)

        # Cache fixed constants for custom continuous-time samplers.
        # log-snr
        self.logsnr_t_min = 0.05
        self.logsnr_t_max = 0.95
        self.logsnr_u_min = 2.0 * math.log(
            self.logsnr_t_min / (1.0 - self.logsnr_t_min)
        )
        self.logsnr_u_max = 2.0 * math.log(
            self.logsnr_t_max / (1.0 - self.logsnr_t_max)
        )
        self.logsnr_t_span = self.logsnr_t_max - self.logsnr_t_min
        # ctsm-is
        self.ctsm_is_t_max = 0.8
        # mle
        self.mle_eps1 = 5e-4
        self.mle_eps2 = 5e-4
        self.mle_rem_eps2 = 1.0 - self.mle_eps2
        self.mle_scale = self.mle_rem_eps2 - self.mle_eps1
        self.mle_L_low = math.log(self.mle_eps1 / (1.0 - self.mle_eps1))
        self.mle_L_high = math.log(self.mle_rem_eps2 / (1.0 - self.mle_rem_eps2))
        self.mle_L_scale = self.mle_L_high - self.mle_L_low

        self.rne = rne
        self.rne_eps1 = eps1 + time_std
        self.rne_eps_factor = 1.0 - eps1 - eps2 - time_std

    def scale_t(self, t):
        # scale t from [0, 1] to [eps1, 1-eps2], or [rne_eps1, 1-eps2-time_std] if rne
        if self.rne:
            return t * self.rne_eps_factor + self.rne_eps1
        else:
            return t * self.eps_factor + self.eps1

    def sample_t(self, batch_size, device):
        if self.t_dist == "uniform":
            return self.scale_t(torch.rand(batch_size, device=device))
        elif self.t_dist == "log-snr":
            # 1. Sample log-SNR uniformly between the cached bounds.
            u = (self.logsnr_u_max - self.logsnr_u_min) * torch.rand(
                batch_size, device=device
            ) + self.logsnr_u_min

            # 2. Transform the sampled log-SNR back into time.
            t = torch.sigmoid(u / 2.0)

            # 3. Normalize t to occupy exactly [0, 1].
            t_norm = (t - self.logsnr_t_min) / self.logsnr_t_span
            t_norm = torch.clamp(t_norm, min=0.0, max=1.0)  # Ensure numerical stability

            # 4. Scale back to the configured interval.
            return self.scale_t(t_norm)
        elif self.t_dist == "ctsm-is":
            u = torch.rand(batch_size, device=device)

            # inverse CDF
            t_norm = (
                (u * self.ctsm_is_t_max)
                / (1.0 + (u - 1.0) * self.ctsm_is_t_max)
                / self.ctsm_is_t_max
            )
            t_norm = torch.clamp(t_norm, min=0.0, max=1.0)

            return self.scale_t(t_norm)
        elif self.t_dist == "mle":
            """
            Samples from p(t) proportional to 1/(t(1-t)) within [eps1, 1-eps2],
            and scales the output back to exactly [0, 1].
            """
            # 1. Sample standard Uniform
            u = torch.rand(batch_size, device=device)

            # 2. Interpolate in logit space using cached bounds.
            y = self.mle_L_low + u * self.mle_L_scale

            # 3. Apply Sigmoid to map back to the [eps1, 1-eps2] space.
            t = torch.sigmoid(y)

            # 4. Scale back to exactly [0, 1].
            t_norm = (t - self.mle_eps1) / self.mle_scale
            t_norm = torch.clamp(t_norm, min=0.0, max=1.0)  # Ensure numerical stability

            return self.scale_t(t_norm)
        else:
            raise NotImplementedError

    def perturb_and_fold_t(self, t, time_std):
        # perturb t with noise of std time_std, then fold back to [eps1, diff_eps2]
        # while ensuring a minimum distance of shift_size from t

        # perturb
        t = t + torch.randn_like(t) * time_std

        # move back to supposedly [0, 1]
        t = (t - self.eps1) / self.eps_factor

        # fold back to [0, 1]
        norm_t = 1.0 - torch.abs((t % 2) - 1.0)

        # move back to [eps1, diff_eps2]
        return self.eps1 + norm_t * self.eps_factor

    def resolve_collisions(self, t, tau):
        """
        Resolves collisions by pushing tau away from t without wrapping.
        shift_size must be >= min_gap.
        """
        min_gap = self.eps
        shift_size = self.eps

        # 1. Identify the collisions
        mask = torch.abs(tau - t) < min_gap

        if mask.any():
            # 2. Determine default push direction based on where tau currently is.
            # If tau drifted right, try to push it further right (+1). Otherwise, left (-1).
            signs = torch.where(tau >= t, 1.0, -1.0)

            # 3. Boundary Overrides (The safety net)
            # If pushing right from `t` would exceed the upper bound, force a left push.
            signs = torch.where(t + shift_size > self.diff_eps2, -1.0, signs)

            # If pushing left from `t` would exceed the lower bound, force a right push.
            signs = torch.where(t - shift_size < self.eps1, 1.0, signs)

            # 4. Apply the shift
            # CRITICAL: We calculate the new position relative to `t`, not `tau`.
            # This absolutely guarantees the distance |tau - t| is exactly `safe_distance`.
            tau = torch.where(mask, t + (signs * shift_size), tau)

        return tau

    def clamp_t(self, t):
        return torch.clamp(t, self.eps1, self.diff_eps2)

    def alpha(self, t):
        return t

    def d_alpha(self, t):
        return torch.ones_like(t)

    def beta(self, t):
        return 1.0 - t

    def d_beta(self, t):
        return -torch.ones_like(t)

    def f(self, tau):
        return -1.0 / (1.0 - tau)

    def g(self, tau):
        sq_gs = 2.0 * tau / (1.0 - tau)
        return torch.sqrt(sq_gs), sq_gs

    def beta_var_d(self, t):
        betas = 1.0 - t
        vars = torch.square(betas)
        d_vars = -2.0 * betas
        return betas, vars, d_vars

    def sample_and_time_score(self, z, t, temp=None):
        # z: clean data
        # t: time step, float between 0 and 1
        # return noisy data, score and score weights
        if temp is not None:
            xs, noises = temp
        else:
            noises = torch.randn_like(z)
            xs = self._expand_batch(t) * z + self._expand_batch(1.0 - t) * noises

        noises_z = torch.sum(noises * z, dim=self.reduce_dims, keepdim=False)
        noises_2 = torch.sum(torch.square(noises), dim=self.reduce_dims, keepdim=False)
        targets = -(self.dim + noises_z - noises_2) / self.dim / self.ctsm_num

        weights = (1.0 - t) / self.dim / self.ctsm_num

        return xs, targets, weights

    def sample_and_velocities(self, z, t, temp=None):
        t = self._expand_batch(t)
        if temp is not None:
            xs, x0 = temp
            vs = (xs - x0) / t
        else:
            x0 = torch.randn_like(z)

            xs = t * z + (1.0 - t) * x0
            vs = z - x0
        return xs, vs

    def score_from_v(self, vs, xs, t):
        # hack: prevent scores being nan
        t = self._expand_batch(t)

        # Calculate the standard scores
        scores = (vs * t - xs) / (1.0 - t)

        # Mask out the scores where t is exactly 0
        scores = torch.where(t == 1.0, torch.zeros_like(scores), scores)

        return scores

    def v_from_score(self, scores, xs, t):
        t = self._expand_batch(t)
        # Assuming that when t == 0, scores are exactly those of standard normal hence scores = -xs
        return torch.where(t > 0.0, xs / t + (1.0 / t - 1.0) * scores, xs)

    def v_from_data_and_noise(self, data, noise, t):
        return data - noise

    def ref_log_prob(self, x, t):
        flat_x = x.view((x.shape[0], -1))
        return gaussian_log_prob(
            torch.zeros_like(flat_x),
            (torch.square(self.alpha(t)) + torch.square(self.beta(t))).view(
                (t.shape[0], 1)
            ),
            flat_x,
        )

    def ref_score(self, x, t):
        orig_shape = x.shape
        flat_x = x.view((x.shape[0], -1))
        return gaussian_score(
            x=flat_x,
            t=t.view((t.shape[0], 1)),
            mus=torch.zeros((flat_x.shape[1]), device=x.device),
            sigma=1.0,
            alpha_fn=self.alpha,
            beta_fn=self.beta,
        ).view(orig_shape)

    def ref_conditional_log_prob(self, x, t, y, tau):
        diff_t = self.to_diffusion_time(t)
        diff_tau = self.to_diffusion_time(tau)
        reverse_mask = diff_tau < diff_t
        forward_mask = ~reverse_mask

        t = self._expand_batch(t)
        diff_t = self._expand_batch(diff_t)
        diff_tau = self._expand_batch(diff_tau)

        delta_t = torch.abs(diff_tau - diff_t)
        std_dt = torch.sqrt(delta_t)

        fs = self.f(diff_t)
        gs, sq_gs = self.g(diff_t)

        stds = gs * std_dt
        means = x.clone()
        # reverse
        if torch.any(reverse_mask):
            scores = self.ref_score(x=x[reverse_mask], t=t[reverse_mask].view(-1))
            means[reverse_mask] = (
                means[reverse_mask]
                - (fs[reverse_mask] * x[reverse_mask] - sq_gs[reverse_mask] * scores)
                * delta_t[reverse_mask]
            )
        # forward
        if torch.any(forward_mask):
            means[forward_mask] = (
                means[forward_mask]
                + fs[forward_mask] * x[forward_mask] * delta_t[forward_mask]
            )

        return self.normal.log_prob(y, means, stds)

    @torch.no_grad()
    def reference(self, x, t, y, tau):
        return (
            self.ref_conditional_log_prob(y, tau, x, t)
            - self.ref_conditional_log_prob(x, t, y, tau)
            + self.ref_log_prob(y, tau)
            - self.ref_log_prob(x, t)
        )


def get_prob_path(config, device, data_statistics, fm_model=None):
    if config.estim.prob_path == "ot":
        return CondOT(
            is_image=config.data.is_image,
            dim=config.data.dim,
            device=device,
            stnce_noise_type=config.estim.stnce_noise_type,
            score_type=config.estim.score_type,
            time_score_type=config.estim.time_score_type,
            data_statistics=data_statistics,
            fm_model=fm_model,
            eps1=config.estim.eps1,
            eps2=config.estim.eps2,
            eps=config.estim.prob_path_eps,
            t_dist=config.estim.t_dist,
            ctsm_c=config.estim.ctsm_c,
            rne=config.estim.rne,
            time_std=config.estim.time_std,
            ei_sde_transition_stats=getattr(
                config.estim, "ei_sde_transition_stats", "original"
            ),
        )

    else:
        raise NotImplementedError
