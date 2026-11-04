import math
import torch
import torch.nn.functional as F


def gaussian_get_posterior_params(means, std, alpha_fn, beta_fn, x, t, tau):
    """
    Computes p(y | x, t, tau) where the prior is a single Gaussian N(mean, std^2 I).

    Args:
        mean (Tensor): Prior mean vector. Shape (D,) or (1, D).
        std (float): Prior standard deviation.
        alpha_fn, beta_fn: Callables returning (B,) given time.
        x (Tensor): Observation at time t. Shape (B, D).
        t, tau (Tensor): Time steps. Shape (B,).
    """
    sigma_sq = std**2
    B, D = x.shape

    # 1. Get schedule params at t
    # alpha, beta: (B,) -> (B, 1) for broadcasting against (B, D)
    alpha = alpha_fn(t).view(B, 1)
    beta = beta_fn(t).view(B, 1)

    # Common denominator: sigma^2 * alpha^2 + beta^2
    denom = sigma_sq * alpha**2 + beta**2  # (B, 1)

    post_var = (sigma_sq * beta**2) / denom  # (B, 1)

    post_mean = (sigma_sq * alpha * x + beta**2 * means) / denom  # (B, D)

    # 4. Project to time tau (Predictive Step)
    alpha_tau = alpha_fn(tau).view(B, 1)
    beta_tau = beta_fn(tau).view(B, 1)

    # Predictive Mean: mu_y = alpha_tau * mu_z
    y_mean = alpha_tau * post_mean  # (B, D)

    # Predictive Variance: var_y = alpha_tau^2 * var_z + beta_tau^2
    y_var = (alpha_tau**2 * post_var) + beta_tau**2  # (B, 1)

    return y_mean, y_var


def gaussian_sample(m, S):
    """
    Sample from N(m, S).
    Args:
        m (Tensor): Mean. Shape (B, D).
        S (Tensor): Variance (scalar). Shape (B, 1).
    """
    noise = torch.randn_like(m)
    std = torch.sqrt(S)  # (B, 1)

    return m + std * noise


def gaussian_log_prob(m, S, y):
    """
    Log probability of y under N(m, S).
    Args:
        m (Tensor): Mean. Shape (B, D).
        S (Tensor): Variance. Shape (B, 1).
        y (Tensor): Target sample. Shape (B, D).
    """
    B, D = m.shape

    # Squared Euclidean distance ||y - m||^2
    diff = y - m
    sq_dist = (diff**2).sum(dim=-1)  # (B,)

    # Remove extra dim from S for calculation
    var = S.squeeze(-1)  # (B,)

    log_p = -0.5 * (sq_dist / var) - 0.5 * D * torch.log(2 * math.pi * var)

    return log_p


def gmm_get_posterior_params(means, std, alpha_fn, beta_fn, x, t, tau, weights=None):
    sigma_sq = std**2
    if weights is None:
        log_weights = torch.log(
            torch.ones(means.shape[0], device=x.device) / means.shape[0]
        )
    else:
        log_weights = torch.log(weights)

    B, D = x.shape

    # 1. Get schedule params and reshape for broadcasting
    # alpha, beta: (B,) -> (B, 1, 1)
    alpha = alpha_fn(t).view(B, 1, 1)
    beta = beta_fn(t).view(B, 1, 1)

    # Common denominator: sigma^2 * alpha^2 + beta^2
    denom = sigma_sq * alpha**2 + beta**2  # (B, 1, 1)

    post_var = (sigma_sq * beta**2) / denom  # (B, 1, 1)

    x_exp = x.unsqueeze(1)
    # means: (K, D) -> (1, K, D)
    means_exp = means.unsqueeze(0)

    # (B, 1, 1)*(B, 1, D) + (B, 1, 1)*(1, K, D) -> (B, K, D)
    post_means = (sigma_sq * alpha * x_exp + beta**2 * means_exp) / denom

    # 4. Posterior Responsibilities (Mixing Weights)
    # We need log p(x | i) = log N(x | alpha * mu_i, marg_var)

    marg_mean = alpha * means_exp  # (B, K, D)
    marg_var = denom  # (B, 1, 1)

    # Squared Euclidean distance: ||x - alpha*mu_i||^2
    # (B, 1, D) - (B, K, D) -> (B, K, D)
    diff = x_exp - marg_mean
    sq_dist = (diff**2).sum(dim=-1)  # (B, K)

    # Log Likelihood of x under component i
    # marg_var is (B, 1, 1), squeeze last dim for (B, 1)
    mv_squeeze = marg_var.squeeze(-1)
    log_marg_likelihood = -0.5 * (sq_dist / mv_squeeze) - 0.5 * D * torch.log(
        2 * math.pi * mv_squeeze
    )

    # Combine with prior weights
    # (1, K) + (B, K) -> (B, K)
    log_resp_unnorm = log_weights.unsqueeze(0) + log_marg_likelihood
    log_resp = F.log_softmax(log_resp_unnorm, dim=1)

    # 2. Get schedule params at tau
    alpha_tau = alpha_fn(tau).view(B, 1, 1)
    beta_tau = beta_fn(tau).view(B, 1, 1)

    # 3. Project Means: mu_y = alpha_tau * mu_z
    y_means = alpha_tau * post_means  # (B, K, D)

    # 4. Project Variance: var_y = alpha_tau^2 * var_z + beta_tau^2
    y_var = (alpha_tau**2 * post_var) + beta_tau**2  # (B, 1, 1)

    return y_means, y_var, log_resp


def gmm_sample(m, S, log_pi):
    # post_means, post_var, log_resp
    B, K, D = m.shape

    # 1. Sample Component Indices
    # (B,) indices in range [0, K-1]
    k_indices = torch.distributions.Categorical(logits=log_pi).sample()

    # 2. Select the specific Mean
    # We need to gather the vector m[b, k_indices[b], :]
    # Expand indices to (B, 1, D)
    k_indices_exp = k_indices.view(B, 1, 1).expand(-1, -1, D)

    # Gather along dimension 1 (K)
    # Result: (B, 1, D) -> (B, D)
    selected_mean = torch.gather(m, 1, k_indices_exp).squeeze(1)

    # 3. Sample
    noise = torch.randn((B, D), device=m.device)  # (B, D)
    std = torch.sqrt(S).squeeze(1)  # (B, 1)

    z_sample = selected_mean + std * noise
    return z_sample


def gmm_log_prob(m, S, log_pi, y):

    B, K, D = m.shape

    # Expand y to (B, 1, D) to compare against K means
    y_exp = y.unsqueeze(1)

    # Squared distance ||y - m_k||^2
    diff = y_exp - m
    sq_dist = (diff**2).sum(dim=-1)  # (B, K)

    # Log Gaussian prob for each component k
    S_squeeze = S.squeeze(-1)  # (B, 1)
    log_gauss_k = -0.5 * (sq_dist / S_squeeze) - 0.5 * D * torch.log(
        2 * math.pi * S_squeeze
    )

    # Final LogSumExp: log \sum_k \pi_k N(z|m_k, S)
    return torch.logsumexp(log_pi + log_gauss_k, dim=1)


def gaussian_score(x, t, mus, sigma, alpha_fn, beta_fn):
    """
    Computes the score \nabla_x log p(x|t) for a single Gaussian distribution.

    Args:
        x (Tensor): Batch of data samples. Shape (B, D).
        t (Tensor): Batch of time steps. Shape (B,) or (B, 1).
        mu (Tensor): Mean of the Gaussian prior. Shape (D,) or (1, D).
        sigma (float): Standard deviation of the Gaussian prior.
        alpha_fn (function): Function taking t, returning tensor shape (B,).
        beta_fn (function): Function taking t, returning tensor shape (B,).

    Returns:
        Tensor: The score vectors. Shape (B, D).
    """
    # Ensure t is the correct shape for broadcasting
    if t.ndim == 1:
        t = t.view(-1, 1)

    # 1. Compute time-dependent coefficients
    # Shapes: (B, 1) to broadcast over D dimensions
    at = alpha_fn(t).view(-1, 1)
    bt = beta_fn(t).view(-1, 1)

    # 2. Compute total variance at time t
    # v_t^2 = alpha_t^2 * sigma^2 + beta_t^2
    vt_sq = (at**2) * (sigma**2) + (bt**2)

    # 3. Compute the mean of x_t
    # mu shape: (D,) -> broadcast to (B, D)
    mean_xt = at * mus

    score = -(x - mean_xt) / vt_sq

    return score


def gaussian_mixture_score(x, t, mus, sigma, alpha_fn, beta_fn):
    """
    Computes the score \nabla_x log p(x|t) for a Gaussian Mixture Model.

    Args:
        x (Tensor): Batch of data samples. Shape (B, D).
        t (Tensor): Batch of time steps. Shape (B,) or (B, 1).
        mus (Tensor): Means of the GMM components. Shape (K, D).
        sigma (float): Standard deviation of the GMM components.
        alpha_fn (function): Function taking t, returning tensor shape (B,).
        beta_fn (function): Function taking t, returning tensor shape (B,).

    Returns:
        Tensor: The score vectors. Shape (B, D).
    """
    # Ensure t is the correct shape for broadcasting
    if t.ndim == 1:
        t = t.view(-1, 1)

    # 1. Compute time-dependent coefficients
    # Shapes: (B, 1, 1) to broadcast over K components and D dimensions
    at = alpha_fn(t).view(-1, 1, 1)
    bt = beta_fn(t).view(-1, 1, 1)

    vt_sq = (at**2) * (sigma**2) + (bt**2)

    x_expanded = x.unsqueeze(1)
    mus_expanded = mus.unsqueeze(0)

    scaled_means = at * mus_expanded
    diff = x_expanded - scaled_means

    squared_dists = torch.sum(diff**2, dim=2)

    # 6. Compute Logits (argument of the softmax)
    # We ignore the constant factors of the Gaussian normalization
    # as they cancel out during the softmax operation.
    # logits = - ||x - mu_k||^2 / (2 * v_t^2)
    logits = -0.5 * squared_dists / vt_sq.squeeze(2)

    # 7. Compute Posterior Weights (gamma_k) using Softmax
    # Shape: (B, K) -> (B, K, 1) for broadcasting
    weights = torch.softmax(logits, dim=1).unsqueeze(2)

    # 8. Compute Weighted Score
    # Gradient of component k is: - (x - alpha_t * mu_k) / vt_sq
    # Score is weighted sum of component gradients.

    component_gradients = -diff / vt_sq

    score = torch.sum(weights * component_gradients, dim=1)

    return score


def gaussian_time_score(x, t, mus, sigma, alpha_fn, beta_fn, d_alpha_fn, d_beta_fn):
    """
    Computes the exact analytical time score partial_t log p_t(x) for a single Gaussian.

    Args:
        x (Tensor): Batch of data samples. Shape (B, D).
        t (Tensor): Batch of time steps. Shape (B,) or (B, 1).
        mus (Tensor): Mean of the Gaussian prior. Shape (D,) or (1, D).
        sigma (float): Standard deviation of the Gaussian prior.
        alpha_fn, beta_fn (callable): Functions taking t, returning tensor shape (B,).
        d_alpha_fn, d_beta_fn (callable): Time derivatives of alpha and beta schedules.

    Returns:
        Tensor: The time score scalars. Shape matches input t.
    """
    if t.ndim == 1:
        t_in = t.view(-1, 1)
    else:
        t_in = t

    D = x.shape[1]

    # 1. Compute time-dependent coefficients and their derivatives
    at = alpha_fn(t_in).view(-1, 1)
    bt = beta_fn(t_in).view(-1, 1)
    dat = d_alpha_fn(t_in).view(-1, 1)
    dbt = d_beta_fn(t_in).view(-1, 1)

    # 2. Compute variance and its time derivative
    vt_sq = (at**2) * (sigma**2) + (bt**2)
    d_vt_sq = 2 * at * dat * (sigma**2) + 2 * bt * dbt

    # 3. Compute spatial difference
    mean_xt = at * mus
    diff = x - mean_xt  # Shape: (B, D)

    # 4. Compute required inner products
    sq_dist = torch.sum(diff**2, dim=1, keepdim=True)  # ||x - alpha * mu||^2
    dot_diff_mu = torch.sum(diff * mus, dim=1, keepdim=True)  # (x - alpha * mu)^T mu

    # 5. Assemble the analytical time derivative terms
    term1 = -0.5 * D * d_vt_sq / vt_sq
    term2 = dat * dot_diff_mu / vt_sq
    term3 = 0.5 * sq_dist * d_vt_sq / (vt_sq**2)

    time_score = term1 + term2 + term3

    return time_score.view_as(t)


def gaussian_mixture_time_score(
    x, t, mus, sigma, alpha_fn, beta_fn, d_alpha_fn, d_beta_fn
):
    """
    Computes the exact analytical time score partial_t log p_t(x) for a GMM.

    Args:
        x (Tensor): Batch of data samples. Shape (B, D).
        t (Tensor): Batch of time steps. Shape (B,) or (B, 1).
        mus (Tensor): Means of the GMM components. Shape (K, D).
        sigma (float): Standard deviation of the GMM components.
        alpha_fn, beta_fn (callable): Functions taking t, returning tensor shape (B,).
        d_alpha_fn, d_beta_fn (callable): Time derivatives of alpha and beta schedules.

    Returns:
        Tensor: The time score scalars. Shape matches input t.
    """
    if t.ndim == 1:
        t_in = t.view(-1, 1)
    else:
        t_in = t

    D = x.shape[1]

    # 1. Compute time-dependent coefficients and derivatives (broadcast for K components)
    at = alpha_fn(t_in).view(-1, 1, 1)
    bt = beta_fn(t_in).view(-1, 1, 1)
    dat = d_alpha_fn(t_in).view(-1, 1, 1)
    dbt = d_beta_fn(t_in).view(-1, 1, 1)

    # 2. Compute variance and its derivative
    vt_sq = (at**2) * (sigma**2) + (bt**2)
    d_vt_sq = 2 * at * dat * (sigma**2) + 2 * bt * dbt

    # Squeeze the D dimension for scalar divisions later -> Shape: (B, 1)
    vt_sq_k = vt_sq.squeeze(2)
    d_vt_sq_k = d_vt_sq.squeeze(2)
    dat_k = dat.squeeze(2)

    # 3. Prepare inputs for broadcasting over K components
    x_expanded = x.unsqueeze(1)  # (B, 1, D)
    mus_expanded = mus.unsqueeze(0)  # (1, K, D)

    # 4. Compute differences and distances
    scaled_means = at * mus_expanded
    diff = x_expanded - scaled_means  # Shape: (B, K, D)

    # ||x - alpha * mu_k||^2 -> Shape: (B, K)
    sq_dists = torch.sum(diff**2, dim=2)

    # (x - alpha * mu_k)^T mu_k -> Shape: (B, K)
    dot_diff_mu = torch.sum(diff * mus_expanded, dim=2)

    # 5. Compute Posterior Weights (w_k) via Softmax (Identical to spatial score)
    logits = -0.5 * sq_dists / vt_sq_k
    weights = torch.softmax(logits, dim=1)  # Shape: (B, K)

    # 6. Compute Analytical Time Score for each individual component k -> Shape: (B, K)
    term1_k = -0.5 * D * d_vt_sq_k / vt_sq_k
    term2_k = dat_k * dot_diff_mu / vt_sq_k
    term3_k = 0.5 * sq_dists * d_vt_sq_k / (vt_sq_k**2)

    comp_time_scores = term1_k + term2_k + term3_k

    # 7. Final Time Score is the weighted sum of component time scores
    time_score = torch.sum(weights * comp_time_scores, dim=1)  # Shape: (B,)

    return time_score.view_as(t)


def time_varying_gmm_log_prob(y, means, sigma, alpha_t, beta_t, num_components):
    """
    Evaluate log p_t(y) for a time-varying isotropic Gaussian mixture.

    If
        x ~ (1 / K) sum_k N(mu_k, sigma^2 I)
    and
        y_t = alpha_t x + beta_t eps,   eps ~ N(0, I),
    then each component of y_t is Gaussian with
        mean_k(t) = alpha_t mu_k
        var(t) = alpha_t^2 sigma^2 + beta_t^2.

    The function supports either a single shared time value for the whole batch or
    one time value per sample.

    Args:
        y: Tensor of shape (N, D), the batch of observations.
        means: Tensor of shape (K, D), the component means.
        sigma: Scalar standard deviation for the isotropic component covariance.
        alpha_t: Scalar, shape (N,), or shape (N, 1) scale factor(s).
        beta_t: Scalar, shape (N,), or shape (N, 1) noise standard deviation(s).
        num_components: Integer K. Must match means.shape[0].

    Returns:
        Tensor of shape (N,) containing log probabilities for each observation.
    """

    N, D = y.shape

    device = y.device
    dtype = y.dtype
    means = means.to(device=device, dtype=dtype)
    sigma = torch.as_tensor(sigma, device=device, dtype=dtype)

    alpha_t = alpha_t.view(N, 1, 1)
    beta_t = beta_t.view(N, 1, 1)

    # Under y_t = alpha_t x + beta_t eps, each component stays Gaussian:
    # N(alpha_t mu_k, (alpha_t^2 sigma^2 + beta_t^2) I).
    component_means = alpha_t * means.unsqueeze(0)  # (N, K, D)
    total_var = alpha_t.square() * sigma.square() + beta_t.square()  # (N, 1, 1)
    total_var = total_var.clamp_min(torch.finfo(dtype).eps)

    sq_dist = (y.unsqueeze(1) - component_means).square().sum(dim=-1)  # (N, K)
    total_var = total_var.squeeze(-1)  # (N, 1)

    log_component_density = -0.5 * (
        sq_dist / total_var + D * (math.log(2.0 * math.pi) + torch.log(total_var))
    )

    return torch.logsumexp(log_component_density, dim=1) - math.log(num_components)
