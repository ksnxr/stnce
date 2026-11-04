"""Contrastive objectives for training EBMs."""

import torch
import torch.nn.functional as F


def perturb_and_fold_t(t, time_std):
    # perturb
    """Perturb times and reflect into [0, 1]; nonpositive width samples uniformly."""
    if time_std > 0.0:
        t = t + torch.randn_like(t) * time_std
    else:
        t = torch.rand_like(t)

    # fold back to [0, 1]
    return 1.0 - torch.abs((t % 2) - 1.0)


def tnce_loss(
    x1: torch.Tensor,
    energy_fn,
    epsilon: float = 0.01,
) -> torch.Tensor:
    """Compute the TNCE loss."""

    x0 = torch.randn_like(x1)
    batch_size, sequence_length = x1.shape
    device = x1.device

    t = torch.rand(batch_size, device=device, dtype=torch.float32)
    t_prime = perturb_and_fold_t(t, epsilon)

    # Sample x_t and x_t'
    t_x = t.unsqueeze(-1)
    t_prime_x = t_prime.unsqueeze(-1)
    x_t = (1 - t_x) * x0 + t_x * x1
    x_t_prime = (1 - t_prime_x) * x0 + t_prime_x * x1

    energy_outputs = energy_fn(
        torch.cat([x_t, x_t_prime, x_t, x_t_prime], dim=0),
        torch.cat([t, t_prime, t_prime, t], dim=0),
    )
    energy = energy_outputs[: 2 * x_t.shape[0]]
    energy_prime = energy_outputs[2 * x_t.shape[0] :]
    classification_loss = 2.0 * F.softplus(-energy_prime + energy)
    classification_loss = classification_loss.mean()

    return classification_loss


def tcnce_loss(
    x1: torch.Tensor,
    energy_fn,
    epsilon: float = 0.01,
) -> torch.Tensor:
    """Compute the TCNCE loss."""
    batch_size, _ = x1.shape
    device = x1.device

    t = torch.rand(batch_size, device=device, dtype=torch.float32)
    t_prime = perturb_and_fold_t(t, 0.0)

    # Sample x_t and x_t'
    t_x = t.unsqueeze(-1)
    t_prime_x = t_prime.unsqueeze(-1)

    all_t_x = torch.concat([t_x, t_prime_x], dim=0)
    all_x1 = torch.concat([x1, x1], dim=0)
    all_x_t = (1 - all_t_x) * torch.randn_like(all_x1) + all_t_x * all_x1

    # Draw a random neighbor of x_t by adding Gaussian noise
    all_x_t_neighbor = all_x_t + epsilon * torch.randn_like(all_x_t)

    energy_outputs = energy_fn(
        torch.cat([all_x_t, all_x_t_neighbor], dim=0),
        torch.cat([all_t_x, all_t_x], dim=0),
    )
    energy_xt = energy_outputs[: all_x_t.shape[0]]
    energy_xt_neighbor = energy_outputs[all_x_t.shape[0] :]

    classification_loss = 2.0 * F.softplus(energy_xt - energy_xt_neighbor)
    classification_loss = classification_loss.mean()

    return classification_loss


def stnce_white_loss(
    x1: torch.Tensor,
    energy_fn,
    epsilon: float = 0.01,
) -> torch.Tensor:
    """Compute the STNCE-style white loss over a batch."""
    x0 = torch.randn_like(x1)
    batch_size, _ = x1.shape
    device = x1.device

    # Sample t strictly inside (0, 1), then perturb it with Gaussian noise.
    t = torch.rand(batch_size, device=device, dtype=torch.float32)
    t_prime = perturb_and_fold_t(t, epsilon)

    t_x = t.unsqueeze(-1)
    t_prime_x = t_prime.unsqueeze(-1)
    x_t = (1 - t_x) * x0 + t_x * x1
    x_t_prime = (1 - t_prime_x) * x0 + t_prime_x * x1

    # Sample a random neighbor of x_t by adding Gaussian noise
    noise = torch.randn_like(x_t)
    x_t_neighbor = x_t + epsilon * noise
    x_t_prime_neighbor = x_t + epsilon * noise

    energy_outputs = energy_fn(
        torch.cat([x_t, x_t_prime, x_t_neighbor, x_t_prime_neighbor], dim=0),
        torch.cat([t, t_prime, t_prime, t], dim=0),
    )

    energy_xt = energy_outputs[: 2 * x_t.shape[0]]
    energy_xt_neighbor = energy_outputs[2 * x_t.shape[0] :]
    logit = energy_xt_neighbor - energy_xt
    loss_per_datapoint = 2.0 * F.softplus(-logit)
    return loss_per_datapoint.mean()
