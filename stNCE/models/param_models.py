import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, MultivariateNormal, MixtureSameFamily
from . import model_utils


@model_utils.register_model(name="param_gaussian")
class ParameterizedGaussian(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.data.dim
        self.device = config.device
        self.dtype = config.dtype
        self.random_init = False
        self.eps = config.data.eps

        self.num_tril = self.dim * (self.dim - 1) // 2
        self.cholesky_offdiag = nn.Parameter(
            torch.zeros(self.num_tril, device=self.device, dtype=self.dtype)
        )
        self.cholesky_diag_unconstrained = nn.Parameter(
            torch.zeros(self.dim, device=self.device, dtype=self.dtype)
        )

        self.log_z = nn.Parameter(torch.zeros((), device=self.device, dtype=self.dtype))

    def get_factor(self):
        L = torch.zeros((self.dim, self.dim), dtype=self.dtype, device=self.device)
        indices = torch.tril_indices(self.dim, self.dim, offset=-1)
        L[indices[0], indices[1]] = self.cholesky_offdiag
        L.diagonal().copy_(F.softplus(self.cholesky_diag_unconstrained) + self.eps)
        return L

    def get_precision(self):
        L = self.get_factor()
        return L @ L.T

    def forward(self, x):
        return (
            1
            / 2
            * torch.einsum("...i,ij,...j->...", x, self.get_precision(), x).unsqueeze(
                dim=-1
            )
            - self.log_z
        )


@model_utils.register_model(name="param_two_gaussians")
class ParameterizedTwoGaussians(nn.Module):
    """
    Two-component Gaussian Mixture Model with:
      - Means: free parameters (2, D)
      - Covariances: Cholesky factor L (Σ = L L^T), diag via softplus for stability
      - Mixture weights: single scalar logit a -> π1 = sigmoid(a), π2 = 1-π1
    forward(x) returns log p(x) with shape (B, 1)
    """

    def __init__(self, config):
        super().__init__()
        self.dim = config.data.dim
        self.device = config.device
        self.dtype = config.dtype
        self.random_init = True
        self.eps = config.data.eps

        # Means
        self.means = nn.Parameter(
            torch.randn((2, self.dim), device=self.device, dtype=self.dtype)
        )

        self.num_tril = self.dim * (self.dim - 1) // 2
        self.cholesky_offdiag = nn.Parameter(
            torch.zeros((2, self.num_tril), device=self.device, dtype=self.dtype)
        )
        self.cholesky_diag_unconstrained = nn.Parameter(
            torch.zeros((2, self.dim), device=self.device, dtype=self.dtype)
        )

        # Mixture single logit (scalar)
        self.logit = nn.Parameter(torch.zeros((), device=self.device, dtype=self.dtype))

    def get_means(self):
        return self.means

    def get_factors(self):
        """
        Build Cholesky factors L_k (k=0,1) with positive diagonals.
        Returns: L shape (2, D, D)
        """
        L = torch.zeros((2, self.dim, self.dim), device=self.device, dtype=self.dtype)
        indices = torch.tril_indices(row=self.dim, col=self.dim, offset=-1)
        L[:, indices[0], indices[1]] = self.cholesky_offdiag
        L.diagonal(dim1=1, dim2=2).copy_(
            F.softplus(self.cholesky_diag_unconstrained) + self.eps
        )
        return L

    def get_weights(self):
        prob = torch.sigmoid(self.logit)
        weights = torch.stack([1.0 - prob, prob], dim=0)
        return weights

    def _dist(self):
        mix = Categorical(probs=self.get_weights())
        comps = MultivariateNormal(loc=self.means, scale_tril=self.get_factors())
        return MixtureSameFamily(mix, comps, validate_args=False)

    def forward(self, x):
        # Match Gaussian model: return (B, 1)
        return -self._dist().log_prob(x).unsqueeze(dim=-1)
