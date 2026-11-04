from typing import Callable, Optional, Tuple
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def _assert_st(s: ArrayLike, t: ArrayLike, *, eps: float):
    assert s.shape == t.shape, "s and t must have the same shape"


def sample_s(
    key: jax.random.PRNGKey,
    t: ArrayLike,
    *,
    eps: float,
    delta_t: float = 1e-4,
) -> jnp.ndarray:
    t = jnp.asarray(t)
    t = jnp.maximum(t, eps + delta_t)  # eps + delta_t <= t <= 1.0
    s = t - delta_t  # eps <= s <= 1.0 - delta_t

    _assert_st(s, t, eps=eps)

    return t, s


def forward_transition(
    key: jax.random.PRNGKey,
    sde,
    xs: ArrayLike,
    t: ArrayLike,
    s: ArrayLike,
    *,
    eps: float = 1e-5,
) -> Tuple[ArrayLike, ArrayLike]:
    _assert_st(s, t, eps=eps)
    z = jax.random.normal(key, xs.shape, dtype=xs.dtype)
    mean, std = sde.forward_transition_params(xs=xs, s=s, delta_t=t - s)
    xt = mean + std * z
    logp = -jnp.sum(z**2, axis=-1, keepdims=True) - xs.shape[-1] * (jnp.log(2.0 * jnp.pi) + 2.0 * jnp.log(std))
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
    _assert_st(s, t, eps=eps)
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
    _assert_st(s, t, eps=eps)
    z = jax.random.normal(key, xt.shape, dtype=xt.dtype)
    mean, std = sde.backward_transition_params(xt=xt, t=t, delta_t=t - s, score_xt=score_xt)
    xs = mean + std * z
    logp = -jnp.sum(z**2, axis=-1, keepdims=True) - xt.shape[-1] * (jnp.log(2.0 * jnp.pi) + 2.0 * jnp.log(std))
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
    _assert_st(s, t, eps=eps)
    mean, std = sde.backward_transition_params(xt=xt, t=t, delta_t=t - s, score_xt=score_xt)
    logp = -jnp.sum(((xs - mean) / std) ** 2, axis=-1, keepdims=True)
    logp -= xt.shape[-1] * (jnp.log(2.0 * jnp.pi) + 2.0 * jnp.log(std))
    logp *= 0.5
    return logp

def log_normal_p(x: ArrayLike, mean: ArrayLike, std: ArrayLike) -> ArrayLike:
    return -0.5 * jnp.sum(
        ((x - mean) / std) ** 2 + jnp.log(2.0 * jnp.pi) + 2.0 * jnp.log(std),
        axis=-1,
        keepdims=True,
    )

def rne_loss_fn(
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
    delta_t: float = 1e-4,
    with_reference: bool = False,
    *,
    eps: float = 1e-5,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    RNE loss adapted to JAX.
    - Samples s and t with fixed gap delta_t (see sample_s).
    - Perturbs x0 to xt for each sampled time.
    - Computes the RNE residual matching loss.
    Args:
      weight: Coefficient for the RNE loss.
      delta_t: Time gap between t and s.
    Returns:
      (weighted_rne_loss, dsm_loss_or_aux) matching other energy regularizers.
    """
    del score_fn  # not used for this loss; log_q_and_score_fn is used below

    BS = x.shape[0]
    key, s_key, noise_key, k_fwd, k_bwd = jax.random.split(key, 5)

    t, s = sample_s(s_key, t, eps=eps, delta_t=delta_t)

    ts = jnp.concatenate([t, s], axis=0)
    x0 = jnp.concatenate([x, x], axis=0)

    mean, std = sde.marginal_prob(x0, ts)

    # TODO: check independent noise
    noise = jax.random.normal(noise_key, mean.shape, dtype=mean.dtype)
    xts = mean + noise * std
    xt = xts[:BS]
    xs = xts[BS:]

    if dsm_loss_fn is not None:
        logp_xts_ts, score_xts_ts = log_q_and_score_fn(xts, features, ts)
        dsm_loss = dsm_loss_fn(noise, score_xts_ts, std, ts)
        logp_xt_t = logp_xts_ts[:BS]
        logp_xs_s = logp_xts_ts[BS:]
        score_xt = score_xts_ts[:BS]
    else:
        dsm_loss = jnp.array(0.0)
        logp_xt_t, score_xt = log_q_and_score_fn(xt, features, t)
        logp_xs_s = log_q_fn(xs, features, s)

    logp_xt_from_xs = forward_transition_logp(k_fwd, sde, xt=xt, xs=xs, t=t, s=s, eps=eps)

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

    residual = logp_xt_t + logp_xs_given_xt - logp_xs_s - logp_xt_from_xs

    if with_reference:

        ref_std_fn = lambda time: (jnp.exp(sde.log_mean_coeff(t)) ** 2 + sde.std(time) ** 2) ** 0.5
        ref_score_fn = lambda x, time: - x / ref_std_fn(time) ** 2

        ref_fwd_mean = xs
        ref_fwd_std = jnp.sqrt(2 * delta_t * s)
        ref_fwd_logp = log_normal_p(xs, jnp.zeros_like(xs), ref_std_fn(t)) + log_normal_p(xt, ref_fwd_mean, ref_fwd_std)

        ref_bwd_mean = xt + ref_score_fn(xt, s) * 2 * t * delta_t
        ref_bwd_std = jnp.sqrt(2 * delta_t * t)
        ref_bwd_logp = log_normal_p(xt, jnp.zeros_like(xt), ref_std_fn(s)) + log_normal_p(xs, ref_bwd_mean, ref_bwd_std)

        residual += ref_fwd_logp - ref_bwd_logp

    per_sample = residual ** 2
    rne_loss = per_sample.mean()

    return weight * rne_loss, dsm_loss
