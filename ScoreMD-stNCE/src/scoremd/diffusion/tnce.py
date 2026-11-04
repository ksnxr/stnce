"""tNCE with optional denoising score matching."""

from typing import Callable, Optional, Tuple
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .time_sampling import sample_s


def tnce_loss_fn(
    key: jax.random.PRNGKey,
    sde,
    dsm_loss_fn: Callable[[ArrayLike, ArrayLike], ArrayLike],
    score_fn: Callable[[ArrayLike, ArrayLike], ArrayLike],
    log_q_fn: Callable[[ArrayLike, ArrayLike], ArrayLike],
    log_q_and_score_fn: Callable[[ArrayLike, ArrayLike], Tuple[ArrayLike, ArrayLike]],
    x: ArrayLike,
    features: Optional[ArrayLike],
    t: ArrayLike,
    weight: float,
    tnce_time_std: float = 0.0,
    *,
    eps: float = 1e-5,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """tNCE with optional DSM regularization.

    Pairs two noise levels using shared Gaussian noise and contrasts time labels.
    Return ``(weight * contrastive_loss, dsm_loss)``; DSM is zero when disabled.
    """
    del score_fn

    BS = x.shape[0]
    key, s_key, noise_key = jax.random.split(key, 3)

    t, s = sample_s(s_key, t, eps=eps, time_std=tnce_time_std)

    ts = jnp.concatenate([t, s], axis=0)
    x0 = jnp.concatenate([x, x], axis=0)
    if features is not None:
        features_ts = jnp.concatenate([features, features], axis=0)
    else:
        features_ts = None

    mean, std = sde.marginal_prob(x0, ts)

    z = jax.random.normal(noise_key, (BS, x.shape[-1]), dtype=x.dtype)
    noise = jnp.concatenate([z, z], axis=0)
    xts = mean + noise * std
    xt = xts[:BS]
    xs = xts[BS:]

    if dsm_loss_fn is not None:
        logp_xts_ts, score_xts = log_q_and_score_fn(xts, features_ts, ts)
        dsm_loss = dsm_loss_fn(noise, score_xts, std, ts)
    else:
        logp_xts_ts = log_q_fn(xts, features_ts, ts)
        dsm_loss = jnp.array(0.0)

    logp_xt_t = logp_xts_ts[:BS]
    logp_xs_s = logp_xts_ts[BS:]

    # Evaluate both cross-time pairs in one batch, ordered as (xs, t), (xt, s).
    logp_cross = log_q_fn(jnp.concatenate([xs, xt], axis=0), features_ts, ts)
    logp_xs_t = logp_cross[:BS]
    logp_xt_s = logp_cross[BS:]

    Z1 = logp_xs_t - logp_xs_s
    Z2 = logp_xt_s - logp_xt_t

    per_sample = jax.nn.softplus(Z1) + jax.nn.softplus(Z2)
    tnce_loss = per_sample.mean()
    return weight * tnce_loss, dsm_loss
