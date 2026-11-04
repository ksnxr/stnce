from typing import Callable, Optional, Tuple
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np


def diffclf_loss_fn(
    key: jax.random.PRNGKey,
    sde,
    dsm_loss_fn: Callable,
    score_fn: Callable,
    log_q_fn: Callable,
    log_q_and_score_fn: Callable,
    x: ArrayLike,
    features: Optional[ArrayLike],
    t: ArrayLike,
    weight: float,
    *,
    k: int = 2,
    eps: float = 1e-5,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Diffusion classification loss (DiffCLF) adapted to JAX.
    - Samples k noise levels per sample.
    - Perturbs x0 to xt for each sampled time.
    - Forms all k x k (i, j) pairs and computes logits = log q_t_i(xt_j).
    - Computes multiclass NLL across time labels per sample and averages.
    - Uses log_q_and_score_fn for diagonal entries to compute DSM loss simultaneously.
    Args:
      weight: Coefficient for the classification term.
    Returns:
      (clf_loss, dsm_loss)
    """
    del score_fn  # not used for classification
    assert k >= 2, "k must be at least 2"

    BS = x.shape[0]
    data_shape = x.shape[1:]
    key, t_key, noise_key = jax.random.split(key, 3)

    # Sample k-1 times uniformly in (eps, 1.0] and concatenate the provided t
    ts_uniform = jax.random.uniform(t_key, (BS, k - 1, 1), minval=eps, maxval=1.0)
    ts = jnp.concatenate([ts_uniform, t.reshape(BS, 1, 1)], axis=1)

    # Perturb x0 -> xt using VP marginals
    x0_expanded = jnp.broadcast_to(x[:, None, ...], (BS, k, *data_shape))
    mean, std = sde.marginal_prob(x0_expanded, ts)
    noise = jax.random.normal(noise_key, mean.shape, dtype=mean.dtype)
    xt = mean + std * noise

    # Diagonal entries: i == j
    xt_diag = xt.reshape(BS * k, *data_shape)
    ts_diag = ts.reshape(BS * k, 1)
    if features is not None:
        features_diag = jnp.repeat(features[:, None, ...], k, axis=1)
        features_diag = features_diag.reshape(BS * k, *features_diag.shape[2:])
    else:
        features_diag = None

    logp_ii, score_ii = log_q_and_score_fn(xt_diag, features_diag, ts_diag)
    logp_ii = logp_ii.reshape(BS, k)

    # Compute DSM loss on the diagonal
    noise_flat = noise.reshape(BS * k, *noise.shape[2:])
    std_flat = std.reshape(BS * k, 1)
    if dsm_loss_fn is not None:
        dsm_loss = dsm_loss_fn(noise_flat, score_ii, std_flat, ts_diag)
    else:
        dsm_loss = jnp.array(0.0)

    # Off-diagonal entries: i != j
    i_idx, j_idx = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
    mask = i_idx != j_idx
    i_idx_off = i_idx[mask]
    j_idx_off = j_idx[mask]

    ts_off = ts[:, i_idx_off, :]
    xt_off = xt[:, j_idx_off, :]

    if features is not None:
        features_off = jnp.repeat(features[:, None, ...], k * (k - 1), axis=1)
        features_off = features_off.reshape(BS * k * (k - 1), *features_off.shape[2:])
    else:
        features_off = None

    ts_off_flat = ts_off.reshape(BS * k * (k - 1), 1)
    xt_off_flat = xt_off.reshape(BS * k * (k - 1), *xt_off.shape[2:])

    logp_off = log_q_fn(xt_off_flat, features_off, ts_off_flat)
    logp_off = logp_off.reshape(BS, k * (k - 1))

    # Reconstruct the full k x k matrix
    logits = jnp.zeros((BS, k, k), dtype=logp_ii.dtype)
    logits = logits.at[:, jnp.arange(k), jnp.arange(k)].set(logp_ii)
    logits = logits.at[:, i_idx_off, j_idx_off].set(logp_off)

    # For each j: lse over i
    lse = jax.scipy.special.logsumexp(logits, axis=1)  # (BS, k)

    # Per-sample multiclass NLL averaged over j, then batch-mean
    clf_per_sample = -(logp_ii - lse).mean(axis=1)  # (BS,)
    clf_loss = clf_per_sample.mean()  # scalar

    return weight * clf_loss, dsm_loss
