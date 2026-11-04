"""stNCE-s with optional denoising score matching."""

from typing import Callable, Optional, Tuple
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .time_sampling import sample_s


def forward_transition(
    key: jax.random.PRNGKey,
    sde,
    xs: ArrayLike,
    t: ArrayLike,
    s: ArrayLike,
    *,
    eps: float = 1e-5,
) -> Tuple[ArrayLike, ArrayLike]:
    z = jax.random.normal(key, xs.shape, dtype=xs.dtype)
    mean, std = sde.forward_transition_params(xs=xs, s=s, delta_t=t - s)
    xt = mean + std * z
    logp = -jnp.sum(z**2, axis=-1, keepdims=True) - xs.shape[-1] * (
        jnp.log(2.0 * jnp.pi) + 2.0 * jnp.log(std)
    )
    logp *= 0.5
    return xt, logp


def forward_transition_logp(
    key: jax.random.PRNGKey,
    sde,
    xt: ArrayLike,
    xs: ArrayLike,
    t: ArrayLike,
    s: ArrayLike,
    *,
    eps: float = 1e-5,
) -> ArrayLike:
    del key  # log-density only; no randomness
    mean, std = sde.forward_transition_params(xs=xs, s=s, delta_t=t - s)
    logp = -jnp.sum(((xt - mean) / std) ** 2, axis=-1, keepdims=True)
    logp -= xs.shape[-1] * (jnp.log(2.0 * jnp.pi) + 2.0 * jnp.log(std))
    logp *= 0.5
    return logp


def backward_transition(
    key: jax.random.PRNGKey,
    sde,
    xt: ArrayLike,
    t: ArrayLike,
    s: ArrayLike,
    score_xt: ArrayLike,
    *,
    eps: float = 1e-5,
) -> Tuple[ArrayLike, ArrayLike]:
    z = jax.random.normal(key, xt.shape, dtype=xt.dtype)
    mean, std = sde.backward_transition_params(
        xt=xt, t=t, delta_t=t - s, score_xt=score_xt
    )
    xs = mean + std * z
    logp = -jnp.sum(z**2, axis=-1, keepdims=True) - xt.shape[-1] * (
        jnp.log(2.0 * jnp.pi) + 2.0 * jnp.log(std)
    )
    logp *= 0.5
    return xs, logp


def backward_transition_logp(
    key: jax.random.PRNGKey,
    sde,
    xt: ArrayLike,
    xs: ArrayLike,
    t: ArrayLike,
    s: ArrayLike,
    score_xt: ArrayLike,
    *,
    eps: float = 1e-5,
) -> ArrayLike:
    del key  # log-density only; no randomness
    mean, std = sde.backward_transition_params(
        xt=xt, t=t, delta_t=t - s, score_xt=score_xt
    )
    logp = -jnp.sum(((xs - mean) / std) ** 2, axis=-1, keepdims=True)
    logp -= xt.shape[-1] * (jnp.log(2.0 * jnp.pi) + 2.0 * jnp.log(std))
    logp *= 0.5
    return logp


def stnce_loss_fn(
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
    stnce_time_std: float = 0.0,
    *,
    eps: float = 1e-5,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """stNCE-s with optional DSM regularization.

    Uses forward-reverse kernel.
    Return ``(weight * contrastive_loss, dsm_loss)``; DSM is zero when disabled.
    """
    del score_fn  # not used for this loss; log_q_and_score_fn is used below

    BS = x.shape[0]
    key, s_key, noise_key, k_fwd, k_bwd = jax.random.split(key, 5)

    t, s = sample_s(s_key, t, eps=eps, time_std=stnce_time_std)


    xs_mean, xs_std = sde.marginal_prob(x, s)

    xs_noise = jax.random.normal(noise_key, x.shape, dtype=x.dtype)
    xs = xs_mean + xs_noise * xs_std
    xt, logp_xt_from_xs = forward_transition(k_fwd, sde, xs=xs, t=t, s=s, eps=eps)

    features_pair = (
        jnp.concatenate([features, features], axis=0)
        if features is not None
        else None
    )

    if dsm_loss_fn is not None:
        ts = jnp.concatenate([t, s], axis=0)
        xts = jnp.concatenate([xt, xs], axis=0)
        xt_mean, xt_std = sde.marginal_prob(x, t)
        xt_noise = (xt - xt_mean) / xt_std

        stds = jnp.concatenate([xt_std, xs_std], axis=0)
        logp_xts_ts, score_xts_ts = log_q_and_score_fn(xts, features_pair, ts)
        dsm_loss = dsm_loss_fn(
            jnp.concatenate([xt_noise, xs_noise]), score_xts_ts, stds, ts
        )
        logp_xt_t = logp_xts_ts[:BS]
        logp_xs_s = logp_xts_ts[BS:]
        score_xt = score_xts_ts[:BS]
    else:
        dsm_loss = jnp.array(0.0)
        logp_xt_t, score_xt = log_q_and_score_fn(xt, features, t)


    logp_xs_given_xt = backward_transition_logp(
        k_fwd,
        sde,
        xt=xt,
        xs=xs,
        t=t,
        s=s,
        score_xt=jax.lax.stop_gradient(score_xt),
        eps=eps,
    )

    xs_proposal, logp_xs_proposal_from_xt = backward_transition(
        k_bwd, sde, xt=xt, t=t, s=s, score_xt=jax.lax.stop_gradient(score_xt), eps=eps
    )
    if dsm_loss_fn is not None:
        logp_xs_proposal = log_q_fn(xs_proposal, features, s)
    else:
        # Both states are evaluated at s; batch their independent energy calls.
        logp_xs_pair = log_q_fn(
            jnp.concatenate([xs, xs_proposal], axis=0),
            features_pair,
            jnp.concatenate([s, s], axis=0),
        )
        logp_xs_s = logp_xs_pair[:BS]
        logp_xs_proposal = logp_xs_pair[BS:]
    logp_xt_given_xs_proposal = forward_transition_logp(
        k_bwd, sde, xt=xt, xs=xs_proposal, t=t, s=s, eps=eps
    )

    Z1 = logp_xt_t + logp_xs_given_xt - logp_xs_s - logp_xt_from_xs

    Z2 = (
        logp_xs_proposal
        + logp_xt_given_xs_proposal
        - logp_xt_t
        - logp_xs_proposal_from_xt
    )

    per_sample = jax.nn.softplus(Z1) + jax.nn.softplus(Z2)
    stnce_loss = per_sample.mean()
    return weight * stnce_loss, dsm_loss
