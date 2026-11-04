import math
import torch
from default import DefaultDistribution
from torch.distributions import (
    Normal,
    Independent,
    MultivariateNormal,
    TransformedDistribution,
)
from torch.distributions.transforms import ReshapeTransform


class GaussiansforMI(DefaultDistribution):
    def __init__(self, config, workdir):
        self.dim = config.data.dim
        self.device = config.device
        self.prefix = f"mi_{self.dim}"
        self.val_set_path = f"val_sets/{self.prefix}_{config.seed}.pt"
        self.test_set_path = f"test_sets/{self.prefix}_{config.seed}.pt"

        self.denom_dist = Independent(
            Normal(
                torch.zeros(self.dim, device=self.device),
                torch.ones(self.dim, device=self.device),
            ),
            reinterpreted_batch_ndims=1,
        )

        self.true_mutual_info = self.get_true_mi()
        self.rho = self.get_rho_from_mi(self.true_mutual_info, self.dim)
        cov_tensor = torch.tensor(
            [[[1, self.rho], [self.rho, 1]] for _ in range(self.dim // 2)],
            device=self.device,
        )
        base_dist = Independent(
            MultivariateNormal(
                loc=torch.zeros(self.dim // 2, 2, device=self.device),
                covariance_matrix=cov_tensor,
            ),
            reinterpreted_batch_ndims=1,
        )
        transform = ReshapeTransform((self.dim // 2, 2), (self.dim,))
        self.dist = TransformedDistribution(base_dist, [transform])

    @staticmethod
    def get_mi_from_rho(self):
        return -0.5 * math.log(1 - self.rho**2) * self.dim

    @staticmethod
    def get_rho_from_mi(mi, n_dims):
        x = (4 * mi) / n_dims
        return (1 - math.exp(-x)) ** 0.5

    def get_true_mi(self):
        if self.dim == 20:
            mi = 5
        elif self.dim == 40:
            mi = 10
        elif self.dim == 80:
            mi = 20
        elif self.dim == 160:
            mi = 40
        elif self.dim == 320:
            mi = 80
        else:
            raise NotImplementedError
        return mi

    def sample(self, num_samples):
        return self.dist.sample((num_samples,))

    def numerator_log_prob(self, x):
        return self.dist.log_prob(x)

    def denominator_log_prob(self, x):
        return self.denom_dist.log_prob(x)

    @torch.no_grad()
    def emprical_mutual_info(self, samples=None):
        if samples is None:
            samples = self.sample(100000)
        return torch.mean(
            self.numerator_log_prob(samples) - self.denominator_log_prob(samples)
        ).cpu()
