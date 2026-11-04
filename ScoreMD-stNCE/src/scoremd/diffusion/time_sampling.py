import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def assert_ordered_time_pair(s: ArrayLike, t: ArrayLike, *, eps: float):
    assert s.shape == t.shape, "s and t must have the same shape"


def resolve_time_collisions(t: ArrayLike, tau: ArrayLike, *, eps: float):
    """Match CondOT.resolve_collisions: push tau away from t by eps."""
    lower = eps
    upper = 1.0 - eps
    mask = jnp.abs(tau - t) < eps

    signs = jnp.where(tau >= t, 1.0, -1.0).astype(t.dtype)
    signs = jnp.where(t + eps > upper, -1.0, signs)
    signs = jnp.where(t - eps < lower, 1.0, signs)

    tau = jnp.where(mask, t + signs * eps, tau)
    return tau


def sample_s(
    key: jax.random.PRNGKey,
    t: ArrayLike,
    *,
    eps: float,
    time_std: float = 0.0,
) -> jnp.ndarray:
    """Sample an ordered time pair shared by TNCE and STNCE."""
    t = jnp.asarray(t)
    if time_std > 0.0:
        s = jax.random.normal(key, shape=t.shape, dtype=t.dtype) * time_std + t

        s = (s - eps) / (1.0 - eps)
        s = 1.0 - jnp.abs((s % 2) - 1.0)
        s = eps + s * (1.0 - eps)
    else:
        s = jax.random.uniform(key, shape=t.shape, minval=eps, maxval=1.0)

    s = resolve_time_collisions(t, s, eps=eps)

    s_new = jnp.minimum(s, t)
    t_new = jnp.maximum(s, t)

    assert_ordered_time_pair(s_new, t_new, eps=eps)

    return t_new, s_new
