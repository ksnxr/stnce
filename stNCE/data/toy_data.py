import os
import math
import torch
import logging
from . import data_utils
from torch.distributions.utils import lazy_property
from torch.distributions import (
    Distribution,
    constraints,
    Normal,
    Categorical,
    Independent,
)
from .default import DefaultDistribution
from utils import build_gmm


def inverse_square_root(m, eps):
    eigvals, eigvecs = torch.linalg.eigh(m)
    new_eigvals = (eigvals.clamp(min=eps)).rsqrt()

    return eigvecs @ torch.diag(new_eigvals) @ eigvecs.T


@data_utils.register_data(name="checkerboard")
class Checkerboard(DefaultDistribution):
    # help from ChatGPT

    def __init__(
        self,
        config,
        workdir,
        m=4,
        D=2,
    ):
        super().__init__(config, workdir, prefix="checkerboard")
        self.dim = 2
        self.device = config.device

        self.m = int(m)
        self.D = torch.as_tensor(D, device=self.device)
        self._valid_squares = [
            (i, j) for i in range(self.m) for j in range(self.m) if (i + j) % 2 == 0
        ]
        self._valid_squares_xy = [
            (-self.D + i * self._cell_size, -self.D + j * self._cell_size)
            for (i, j) in self._valid_squares
        ]
        self._valid_squares_center = [
            (
                -self.D + (i + 0.5) * self._cell_size,
                -self.D + (j + 0.5) * self._cell_size,
            )
            for (i, j) in self._valid_squares
        ]
        self._n_black = len(self._valid_squares)
        self._valid_squares = torch.tensor(self._valid_squares, device=self.device)
        self._valid_squares_xy = torch.tensor(
            self._valid_squares_xy, device=self.device
        )
        self._valid_squares_center = torch.tensor(
            self._valid_squares_center, device=self.device
        )
        self.plot_lims = (-D - 0.5, D + 0.5)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @lazy_property
    def _cell_size(self):
        return (2.0 * self.D) / self.m  # side length of one checker square

    @lazy_property
    def _log_const(self):
        # log density value on a black square: -log(area * #black)
        return -math.log(self._n_black) - 2.0 * math.log(self._cell_size)

    # ------------------------------------------------------------------ #
    # sampling
    # ------------------------------------------------------------------ #
    def sample(self, num_samples):
        square_indices = torch.randint(
            0, self._n_black, size=(num_samples,), device=self.device
        )
        selected_xys = self._valid_squares_xy[square_indices]
        offsets = (
            torch.rand(size=(num_samples, 2), device=self.device) * self._cell_size
        )
        return selected_xys + offsets

    # ------------------------------------------------------------------ #
    # log-density evaluation
    # ------------------------------------------------------------------ #
    def log_prob(self, x):
        s = self._cell_size

        # convert coordinates to non-negative cell indices
        col = torch.floor((x[..., 0] + self.D) / s).to(torch.long)
        row = torch.floor((x[..., 1] + self.D) / s).to(torch.long)
        in_unit_square = (col >= 0) & (col < self.m) & (row >= 0) & (row < self.m)

        is_black = ((row + col) % 2 == 0) & in_unit_square
        out = torch.full((x.shape[:-1], 1), float("-inf"), device=self.device)
        out[is_black, :] = self._log_const
        return out

    # ------------------------------------------------------------------ #
    @lazy_property
    def mean(self):
        return torch.mean(self._valid_squares_center, dim=0)

    @lazy_property
    def variance(self):
        var1 = torch.var(self._valid_squares_center, dim=0, unbiased=False)
        # variance inside squares
        var2 = self._cell_size**2 / 12
        return var1 + var2


@data_utils.register_data(name="eight_gaussians")
class EightGaussians(DefaultDistribution):
    def __init__(self, config, workdir):
        super().__init__(config, workdir, prefix="eight_gaussians")
        self.dim = 2
        assert config.data.dim == 2
        self.device = config.device

        temp = 1.0 / math.sqrt(2.0)
        self.mus = 2.0 * torch.tensor(
            [
                [-1.0, 0.0],
                [-temp, -temp],
                [0.0, -1.0],
                [temp, -temp],
                [1.0, 0.0],
                [temp, temp],
                [0.0, 1.0],
                [-temp, temp],
            ],
            device=self.device,
        )
        self.sigma = 0.5

        self.dist = build_gmm(
            num_components=8,
            dim=self.dim,
            mus=self.mus,
            sigma=self.sigma,
            device=self.device,
        )
        self.plot_lims = (-2.5, 2.5)

    def sample(self, num_samples):
        return self.dist.sample((num_samples,))

    def log_prob(self, x):
        return self.dist.log_prob(x).unsqueeze(dim=-1)

    def get_data_statistics(self):
        return {"means": self.mus, "std": self.sigma}


@data_utils.register_data(name="distant_eight_gaussians")
class DistantEightGaussians(DefaultDistribution):
    def __init__(self, config, workdir):
        super().__init__(config, workdir, prefix="distant_eight_gaussians")
        self.dim = 2
        assert config.data.dim == 2
        self.device = config.device

        temp = 1.0 / math.sqrt(2.0)
        self.mus = 5.0 * torch.tensor(
            [
                [-1.0, 0.0],
                [-temp, -temp],
                [0.0, -1.0],
                [temp, -temp],
                [1.0, 0.0],
                [temp, temp],
                [0.0, 1.0],
                [-temp, temp],
            ],
            device=self.device,
        )
        self.sigma = 0.5

        self.dist = build_gmm(
            num_components=8,
            dim=self.dim,
            mus=self.mus,
            sigma=self.sigma,
            device=self.device,
        )
        self.plot_lims = (-6.0, 6.0)

    def sample(self, num_samples):
        return self.dist.sample((num_samples,))

    def log_prob(self, x):
        return self.dist.log_prob(x).unsqueeze(dim=-1)

    def get_data_statistics(self):
        return {"means": self.mus, "std": self.sigma}


@data_utils.register_data(name="interp_distant_eight_gaussians")
class InterpDistantEightGaussians(DefaultDistribution):
    def __init__(self, config, workdir):
        super().__init__(config, workdir, prefix="interp_distant_eight_gaussians")
        self.dim = 2
        assert config.data.dim == 2
        self.device = config.device

        interp_alpha1 = config.data.interp_alpha1
        interp_alpha2 = config.data.interp_alpha2

        temp = 1.0 / math.sqrt(2.0)
        mus = 5.0 * torch.tensor(
            [
                [-1.0, 0.0],
                [-temp, -temp],
                [0.0, -1.0],
                [temp, -temp],
                [1.0, 0.0],
                [temp, temp],
                [0.0, 1.0],
                [-temp, temp],
            ],
            device=self.device,
        )
        sigma = 0.5

        mus1 = mus * interp_alpha1
        sigma1 = math.sqrt(sigma**2 + (1.0 - interp_alpha1**2))

        self.base_dist = build_gmm(
            num_components=8, dim=self.dim, mus=mus1, sigma=sigma1, device=self.device
        )

        mus2 = mus * interp_alpha2
        sigma2 = math.sqrt(sigma**2 + (1.0 - interp_alpha2**2))

        self.dist = build_gmm(
            num_components=8,
            dim=self.dim,
            mus=mus2,
            sigma=sigma2,
            device=self.device,
        )
        self.plot_lims = (-6.0, 6.0)

    def sample(self, num_samples):
        return self.dist.sample((num_samples,))

    def log_prob(self, x):
        return self.dist.log_prob(x).unsqueeze(dim=-1)


@data_utils.register_data(name="grid_gaussians")
class GridGaussians(DefaultDistribution):
    def __init__(self, config, workdir):
        super().__init__(config, workdir, prefix="grid_gaussians")
        self.dim = 2
        assert config.data.dim == 2
        self.device = config.device

        centers = torch.tensor(
            [[x / 2, y / 2] for x in range(-2, 3) for y in range(-2, 3)],
            device=self.device,
            dtype=torch.float,
        )

        self.mus = centers
        self.sigma = 0.1

        self.dist = build_gmm(
            num_components=16,
            dim=self.dim,
            mus=self.mus,
            sigma=self.sigma,
            device=self.device,
        )
        self.plot_lims = (-2.3, 2.3)

    def sample(self, num_samples):
        return self.dist.sample((num_samples,))

    def log_prob(self, x):
        return self.dist.log_prob(x).unsqueeze(dim=-1)

    def get_data_statistics(self):
        return {"means": self.mus, "std": self.sigma}


class RingTruncNormal2D(Distribution):
    r"""
    2D 'ring' distribution:
        R ~ TruncatedNormal(mu, sigma^2; support r>=0)
        Θ ~ Uniform[0, 2π), independent
        (X,Y) = (R cos Θ, R sin Θ)

    log_prob(x,y) = log p_R(r) - log(2π) - log r,  r = sqrt(x^2+y^2).

    Parameters may broadcast; resulting samples have event_shape=(2,).
    """

    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.real_vector
    has_rsample = False  # sampling not reparameterized

    def __init__(self, loc, scale, device, *, validate_args=None):
        self.loc = torch.tensor(loc, device=device)
        self.scale = torch.tensor(scale, device=device)
        self.device = device
        super().__init__(
            batch_shape=torch.Size([1]),
            event_shape=torch.Size([2]),
            validate_args=validate_args,
        )

        # standard normal helper
        self._std = Normal(
            torch.zeros(1, device=self.device), torch.ones(1, device=self.device)
        )

        # CDF at lower bound 0: α = Φ((0-μ)/σ) = Φ(-μ/σ)
        self._alpha = self._std.cdf(
            (0.0 - self.loc) / self.scale
        )  # same shape as params
        # log(1-α) stable
        self._log_norm = torch.log1p(-self._alpha)  # denominator for truncation

        # precompute log σ and const
        self._log_scale = torch.log(self.scale)
        self._half_log2pi = 0.5 * math.log(2.0 * math.pi)
        self._log2pi = math.log(2.0 * math.pi)

    # ------------------------------------------------------------------ #
    def _sample_radius(self, num_samples):
        """
        Inverse-CDF sampling from lower-truncated Normal at 0.
        """
        # U ~ Uniform[0,1]
        u = torch.rand((num_samples,), device=self.device)
        # Map to truncated Normal CDF range: α + (1-α)u
        p = self._alpha + (1.0 - self._alpha) * u
        # Invert standard normal CDF, then scale/shift
        z = self._std.icdf(p)  # standard quantile
        r = self.loc + self.scale * z  # back to trunc-normal
        # numerical guard: clamp to >=0
        return torch.clamp(r, min=0.0)

    # ------------------------------------------------------------------ #
    def sample(self, num_samples):
        r = self._sample_radius(num_samples)  # (...batch)
        theta = 2.0 * math.pi * torch.rand_like(r)
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        return torch.stack((x, y), dim=-1)  # (...batch, 2)

    # ------------------------------------------------------------------ #
    def log_prob(self, value: torch.Tensor):
        """
        value: (..., 2)
        returns: (...,) broadcast over batch_shape
        """

        # radius
        r = value.norm(dim=-1)  # (...,)

        # log p_R(r) for r>=0
        z = (r - self.loc) / self.scale
        log_phi = -0.5 * z.pow(2) - self._half_log2pi  # log standard normal pdf
        log_pR = log_phi - self._log_scale - self._log_norm

        # add angular + Jacobian terms
        logp = log_pR - self._log2pi - torch.log(r)

        # outside support: r<0 (can't happen from norm) but guard r==0
        logp = torch.where(r > 0, logp, torch.full_like(logp, float("-inf")))
        return logp


class MyMixtureDistribution(Distribution):
    def __init__(self, dim, mixture_log_probs, mixture, components, device):
        self.dim = dim
        self.mixture_log_probs = mixture_log_probs
        self.mixture = mixture
        self.components = components
        self.device = device

    def sample(self, num_samples):
        if isinstance(num_samples, (tuple, list)):
            num_samples = num_samples[0]
        idxes = self.mixture.sample((num_samples,))
        output = torch.zeros((num_samples, self.dim), device=self.device)
        for k, comp in enumerate(self.components):
            mask = idxes == k
            if mask.any():
                num = mask.sum().item()
                comp_samples = comp.sample(num)
                output[mask] = comp_samples
        return output

    def log_prob(self, x):
        comp_lps = torch.stack([comp.log_prob(x) for comp in self.components], dim=0)
        lps = torch.logsumexp(self.mixture_log_probs.view((-1, 1)) + comp_lps, dim=0)
        return lps


@data_utils.register_data(name="rings")
class Rings(DefaultDistribution):
    def __init__(self, config, workdir):
        super().__init__(config, workdir, prefix="rings")
        self.dim = 2
        assert config.data.dim == 2
        self.device = config.device

        self.mixture_probs = torch.tensor([0.6, 0.2, 0.2], device=self.device)
        self.mixture_log_probs = torch.log(self.mixture_probs)
        self.mixture = Categorical(probs=self.mixture_probs)
        scale = 5e-2
        self.components = [
            RingTruncNormal2D(loc=0.5, scale=scale, device=self.device),
            RingTruncNormal2D(loc=1.0, scale=scale, device=self.device),
            RingTruncNormal2D(loc=1.5, scale=scale, device=self.device),
        ]
        self.dist = MyMixtureDistribution(
            dim=self.dim,
            mixture_log_probs=self.mixture_log_probs,
            mixture=self.mixture,
            components=self.components,
            device=self.device,
        )
        self.plot_lims = (-2.0, 2.0)

    def sample(self, num_samples):
        if isinstance(num_samples, (tuple, list)):
            num_samples = num_samples[0]
        return self.dist.sample(num_samples)

    def log_prob(self, x):
        return self.dist.log_prob(x).unsqueeze(dim=-1)


@data_utils.register_data(name="simple_gmm")
class SimpleGMM(DefaultDistribution):
    def __init__(self, config, workdir):
        super().__init__(config, workdir, prefix=f"simple_gmm")
        self.dim = config.data.dim
        self.device = config.device
        assert self.dim == 2

        num_components = 2
        self.mus = torch.tensor([[1.0, 1.0], [-1.0, -1.0]], device=self.device)
        self.sigma = 0.3

        self.dist = build_gmm(
            num_components=num_components,
            dim=self.dim,
            mus=self.mus,
            sigma=self.sigma,
            device=self.device,
        )

        # Set plot limits for visualization (mainly useful for 2D)
        self.plot_lims = (-1.6, 1.6)

    def sample(self, num_samples):
        return self.dist.sample((num_samples,))

    def log_prob(self, x):
        return self.dist.log_prob(x).unsqueeze(dim=-1)

    def get_data_statistics(self):
        return {"means": self.mus, "std": self.sigma}


@data_utils.register_data(name="grid_gmm")
class GridGMM(DefaultDistribution):
    def __init__(self, config, workdir):
        super().__init__(config, workdir, prefix=f"grid_gmm_{config.data.sigma}")
        self.dim = config.data.dim
        self.sigma = config.data.sigma  # original: 0.1
        self.device = config.device
        assert self.dim >= 2

        # Create 2*D + 1 components:
        # 1 at origin, 2 for each dimension at +/-1
        num_components = 2 * self.dim + 1

        # Create means: [0,0,...,0] and [0,...,±1,...,0] for each dimension
        mus = torch.zeros((num_components, self.dim), device=self.device)
        idx = 1
        for d in range(self.dim):
            # Add +1 component
            mus[idx, d] = 1.0
            idx += 1
            # Add -1 component
            mus[idx, d] = -1.0
            idx += 1

        self.means = mus

        self.dist = build_gmm(
            num_components=num_components,
            dim=self.dim,
            mus=self.means,
            sigma=self.sigma,
            device=self.device,
        )

        # Set plot limits for visualization (mainly useful for 2D)
        self.plot_lims = (-1.5, 1.5)

    def sample(self, num_samples):
        return self.dist.sample((num_samples,))

    def log_prob(self, x):
        return self.dist.log_prob(x).unsqueeze(dim=-1)

    def get_data_statistics(self):
        return {
            "means": self.means,
            "std": self.sigma,
        }


@data_utils.register_data(name="random_gmm")
class RandomGMM(DefaultDistribution):
    def __init__(self, config, workdir):
        super().__init__(
            config,
            workdir,
            prefix=f"random_gmm_{config.data.num_components}_{config.data.sigma}_{config.data.mu_scale}",
        )
        self.dim = config.data.dim
        self.num_components = config.data.num_components
        self.sigma = config.data.sigma  # original: 0.1
        mu_scale = config.data.mu_scale
        self.device = config.device
        assert self.dim >= 2

        save_path = f"data_cache/random_gmm_{self.dim}_{self.num_components}.pt"
        if config.training.setup:
            assert config.seed == 1, "Only seed 1 is supported for initialization."
            if os.path.exists(save_path):
                logging.info(f"Removing existing {save_path} for reinitialization.")
                os.remove(save_path)
            mus = torch.randn((self.num_components, self.dim), device=self.device)
            torch.save(mus, save_path)
        else:
            mus = torch.load(save_path, map_location=self.device)
        self.means = mus * mu_scale

        self.dist = build_gmm(
            num_components=self.num_components,
            dim=self.dim,
            mus=self.means,
            sigma=self.sigma,
            device=self.device,
        )

        # Set plot limits for visualization (mainly useful for 2D)
        self.plot_lims = (-1.5, 1.5)

    def sample(self, num_samples):
        return self.dist.sample((num_samples,))

    def log_prob(self, x):
        return self.dist.log_prob(x).unsqueeze(dim=-1)

    def get_data_statistics(self):
        return {
            "means": self.means,
            "std": self.sigma,
        }


@data_utils.register_data(name="distant_gaussian")
class DistantGaussian(DefaultDistribution):
    def __init__(self, config, workdir):
        super().__init__(config, workdir, prefix="distant_gaussian")
        self.dim = config.data.dim
        self.device = config.device
        self.means = torch.full((self.dim,), 10.0, device=self.device)
        stds = torch.ones(self.dim, device=self.device)
        self.dist = Independent(Normal(self.means, stds), 1)

        self.plot_lims = (7.0, 13.0)

    def sample(self, num_samples):
        return self.dist.sample((num_samples,))

    def log_prob(self, x):
        return self.dist.log_prob(x).unsqueeze(dim=-1)

    def get_data_statistics(self):
        return {
            "means": self.means,
            "std": 1.0,
        }


@data_utils.register_data(name="two_distant_gaussians")
class TwoDistantGaussians(DefaultDistribution):
    def __init__(self, config, workdir):
        super().__init__(config, workdir, prefix="two_distant_gaussians")
        self.dim = config.data.dim
        self.device = config.device
        self.means = torch.stack(
            [
                torch.full((self.dim,), 1.5, device=self.device),
                torch.full((self.dim,), -1.5, device=self.device),
            ],
            dim=0,
        )
        self.std = 0.1
        self.dist = build_gmm(
            num_components=2,
            dim=self.dim,
            mus=self.means,
            sigma=self.std,
            device=self.device,
        )

        self.plot_lims = (-2.0, 2.0)

    def sample(self, num_samples):
        return self.dist.sample((num_samples,))

    def log_prob(self, x):
        return self.dist.log_prob(x).unsqueeze(dim=-1)

    def get_data_statistics(self):
        return {
            "means": self.means,
            "std": self.std,
        }
