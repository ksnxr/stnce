"""Construct contrastive samples and regression targets."""

import torch

from utils import MyGaussian
from prob_paths_lib import VP


def get_nce_noise(config):
    dim = config.data.dim
    std = config.estim.nce_std
    nce_nu = config.estim.nce_nu

    dist = MyGaussian(dim=dim, std=std)

    def log_noise_fn(x):
        return dist.log_prob(x)

    def build_nce_noise_fn(original_data, train=True, model_energy_fn=None):
        noise = std * torch.randn(
            (original_data.shape[0] * nce_nu, dim),
            device=original_data.device,
        )

        return (original_data, noise, log_noise_fn)

    return build_nce_noise_fn


def get_cnce_noise(config):
    std = config.estim.cnce_std
    cnce_nu = config.estim.cnce_nu

    def build_cnce_noise_fn(original_data, train=True, model_energy_fn=None):
        conditional_data = torch.repeat_interleave(
            original_data, repeats=cnce_nu, dim=0
        )
        conditional_data = conditional_data + std * torch.randn_like(conditional_data)
        cond = torch.zeros(
            (original_data.shape[0] * cnce_nu, 1), device=original_data.device
        )
        return (original_data, conditional_data, cond)

    return build_cnce_noise_fn


def get_stnce_noise(config):
    dim = config.data.dim
    nce_std = config.estim.nce_std
    cnce_std = config.estim.cnce_std
    nce_nu = config.estim.nce_nu
    cnce_nu = config.estim.cnce_nu

    dist = MyGaussian(dim=dim, std=nce_std)

    def log_noise_fn(x):
        return dist.log_prob(x)

    def build_stnce_noise_fn(original_data, train=True, model_energy_fn=None):

        conditional_noise = torch.repeat_interleave(
            original_data, repeats=cnce_nu, dim=0
        )
        conditional_noise = conditional_noise + cnce_std * torch.randn_like(
            conditional_noise
        )
        data_cond = torch.zeros(
            (conditional_noise.shape[0], 1), device=original_data.device
        )

        noise = nce_std * torch.randn(
            (original_data.shape[0] * nce_nu, dim), device=original_data.device
        )
        conditional_data = torch.repeat_interleave(noise, repeats=cnce_nu, dim=0)
        conditional_data = conditional_data + cnce_std * torch.randn_like(
            conditional_data
        )
        noise_cond = torch.zeros(
            (conditional_data.shape[0], 1), device=original_data.device
        )

        return (
            original_data,
            conditional_noise,
            data_cond,
            noise,
            conditional_data,
            noise_cond,
            log_noise_fn,
        )

    return build_stnce_noise_fn


def get_cont_times(
    batch_size,
    time_std,
    prob_path,
    device,
    ts=None,
    rne=False,
    resolve_collisions=False,
):
    if ts is None:
        ts = prob_path.sample_t(batch_size, device=device)

    if time_std == -1.0:
        taus = prob_path.sample_t(batch_size, device=device)
        if resolve_collisions:
            taus = prob_path.resolve_collisions(ts, taus)
    else:
        if rne:
            taus = prob_path.clamp_t(ts - time_std)
        else:
            taus = prob_path.perturb_and_fold_t(ts, time_std)
            if resolve_collisions:
                taus = prob_path.resolve_collisions(ts, taus)
    return ts, taus


def get_telescoping_noise(config, prob_path):
    num_noise_classes = config.estim.num_noise_classes
    num_cla_classes = config.estim.num_cla_classes
    rne = config.estim.rne
    resolve_collisions = config.estim.resolve_collisions

    if num_noise_classes == float("inf"):
        time_std = config.estim.time_std

        if num_cla_classes == 2:

            def build_noise_fn(
                original_data, train=False, temp=None, model_energy_fn=None
            ):
                if temp is not None:
                    xs, ts, _ = temp
                else:
                    xs = None
                    ts = None

                batch_size = original_data.shape[0]

                ts, taus = get_cont_times(
                    batch_size=batch_size,
                    time_std=time_std,
                    prob_path=prob_path,
                    device=original_data.device,
                    ts=ts,
                    rne=rne,
                    resolve_collisions=resolve_collisions,
                )

                if xs is None:
                    xs = prob_path.sample(original_data, ts)

                if temp is None:
                    return xs, ts, taus
                else:
                    return xs, taus

        else:
            raise NotImplementedError

        return build_noise_fn
    else:
        # discrete time
        num_noise_classes = int(num_noise_classes)
        num_classes = num_noise_classes + 1
        prob_path = VP(
            num_steps=num_classes,
            is_image=config.data.is_image,
            dim=config.data.dim,
            device=config.device,
        )
        normalize_time = (
            config.estim.param == "energy"
        ) and config.estim.normalize_time

        if num_cla_classes == 2:
            # drop divisble by 2 constraint
            def build_noise_fn(original_data, train=False):
                # xs: p_{t+1}, ys: p_{t}
                batch_size = original_data.shape[0]
                times = torch.randint(
                    low=0,
                    high=num_noise_classes,
                    size=(batch_size,),
                    device=original_data.device,
                )
                xs = prob_path.sample(original_data, times + 1)
                ys = prob_path.sample(original_data, times)
                if normalize_time:
                    times = times / num_noise_classes
                return (xs, ys, times)

        else:
            # drop divisble by 2 constraint
            def build_noise_fn(original_data, train=False):
                # xs: p_{t+1}, ys: p_{t}
                batch_size = original_data.shape[0]
                times = torch.randint(
                    low=0,
                    high=num_noise_classes + 1,
                    size=(batch_size,),
                    device=original_data.device,
                )
                xs = prob_path.sample(original_data, times)
                if normalize_time:
                    times = times / num_noise_classes
                return (xs, times)

        return build_noise_fn


def get_telescoping_c_noise(config, prob_path):
    # let us optimize it later
    num_noise_classes = config.estim.num_noise_classes

    if num_noise_classes == float("inf"):
        noise_type = config.estim.cnce_noise_type
        assert noise_type.startswith("white")
        noise_std = float(noise_type.split("-")[-1])

        def build_noise_fn(original_data, train=False, temp=None, model_energy_fn=None):
            batch_size = original_data.shape[0]

            if temp is not None:
                xs, ts, _ = temp
            else:
                xs = None
                ts = prob_path.sample_t(batch_size, device=original_data.device)

            if xs is None:
                xs = prob_path.sample(original_data, ts)
            ys = xs + torch.randn_like(xs) * noise_std

            conds = torch.zeros_like(ts)  # symmetric kernel hence cancal out

            taus = ts

            if temp is None:
                return (xs, ts, ys, taus, conds)
            else:
                return (ys, taus, conds)

        return build_noise_fn
    else:
        raise NotImplementedError


def get_telescoping_st_noise(config, prob_path):
    # let us optimize it later
    num_noise_classes = config.estim.num_noise_classes

    if num_noise_classes == float("inf"):
        rne = config.estim.rne
        rne_reference = config.estim.rne_reference
        time_std = config.estim.time_std
        resolve_collisions = config.estim.resolve_collisions

        def build_noise_fn(original_data, train=False, temp=None, model_energy_fn=None):
            batch_size = original_data.shape[0]

            if temp is not None:
                xs, ts, _ = temp
            else:
                xs = None
                ts = None

            ts, taus = get_cont_times(
                batch_size=batch_size,
                time_std=time_std,
                prob_path=prob_path,
                device=original_data.device,
                ts=ts,
                rne=rne,
                resolve_collisions=resolve_collisions,
            )

            if xs is None:
                xs = prob_path.sample(original_data, ts)

            def get_cond_scores(x, t, reverse_mask):
                if (prob_path.score_type == "none") or (not torch.any(reverse_mask)):
                    return None
                if prob_path.score_type == "self":
                    if model_energy_fn is None:
                        raise ValueError(
                            "model_energy_fn is required when score_type='self'."
                        )
                    with torch.enable_grad():
                        temp_x = x[reverse_mask].detach().requires_grad_(True)
                        return -torch.autograd.grad(
                            model_energy_fn(temp_x, t[reverse_mask]).sum(),
                            temp_x,
                            create_graph=False,
                        )[0].detach()
                return prob_path.get_target_score(
                    x=x[reverse_mask], t=t[reverse_mask]
                ).detach()

            xs_scores = get_cond_scores(xs, ts, taus > ts)
            ys = prob_path.conditional_sample(xs, ts, taus, scores=xs_scores)
            conds1 = prob_path.conditional_log_prob(xs, ts, ys, taus, scores=xs_scores)

            ys_scores = get_cond_scores(ys, taus, ts > taus)
            conds2 = prob_path.conditional_log_prob(ys, taus, xs, ts, scores=ys_scores)

            conds = conds1 - conds2
            if rne and rne_reference:
                conds = conds + prob_path.reference(xs, ts, ys, taus)

            if temp is None:
                return (xs, ts, ys, taus, conds.unsqueeze(dim=-1))
            else:
                return (ys, taus, conds.unsqueeze(dim=-1))

        return build_noise_fn

    else:
        raise NotImplementedError


def get_dsm_noise(config, prob_path):
    """Build DSM targets and weights; optional temp reuses (x, t, noise)."""
    num_noise_classes = config.estim.num_noise_classes

    if num_noise_classes == float("inf"):

        def build_dsm_noise_fn(
            original_data, train=False, temp=None, model_energy_fn=None
        ):
            if temp is not None:
                xs, ts, noises = temp
            else:
                ts = prob_path.sample_t(
                    original_data.shape[0], device=original_data.device
                )
                xs, noises = prob_path.sample_and_noises(original_data, ts)

            xs, targets, weights = prob_path.sample_and_stein_score(
                original_data, ts, [xs, noises]
            )
            if temp is None:
                return xs, ts, targets, weights
            else:
                return targets, weights

        return build_dsm_noise_fn

    else:
        raise NotImplementedError


def get_fm_noise(config, prob_path):
    """Build FM targets, optionally reusing (x, t, noise)."""

    def build_noise_fn(original_data, train=False, temp=None, model_energy_fn=None):
        if temp is not None:
            xs, ts, noises = temp
            vs = prob_path.v_from_data_and_noise(original_data, noises, ts)
        else:
            ts = prob_path.sample_t(original_data.shape[0], device=original_data.device)
            xs, vs = prob_path.sample_and_velocities(original_data, ts)
        if temp is None:
            return xs, ts, vs
        else:
            return ts, vs

    return build_noise_fn


def get_ctsm_noise(config, prob_path):
    """Build CTSM targets and weights for continuous-time paths."""
    num_noise_classes = config.estim.num_noise_classes

    if num_noise_classes == float("inf"):

        def build_tsm_noise_fn(
            original_data, train=False, temp=None, model_energy_fn=None
        ):
            if temp is not None:
                xs, ts, noises = temp
            else:
                ts = prob_path.sample_t(
                    original_data.shape[0], device=original_data.device
                )
                xs, noises = prob_path.sample_and_noises(original_data, ts)

            xs, targets, weights = prob_path.sample_and_time_score(
                original_data, ts, [xs, noises]
            )

            if temp is None:
                return xs, ts, targets, weights
            else:
                return targets, weights

        return build_tsm_noise_fn

    else:
        raise NotImplementedError


def get_reuse_main_noise(config, prob_path, estimator):

    assert config.estim.num_noise_classes == float("inf")
    assert config.estim.param == "energy"

    assert config.estim.num_cla_classes == 2
    if estimator == "tcnce":
        noise_type = config.estim.cnce_noise_type
        assert noise_type.startswith("white")
        noise_std = float(noise_type.split("-")[-1])
    elif estimator == "tstnce":
        noise_std = prob_path.std

    time_std = config.estim.time_std
    rne = config.estim.rne
    resolve_collisions = config.estim.resolve_collisions
    use_scores = prob_path.stnce_noise_type in ["diff", "inf-diff"]

    def get_scores(x, t, model_energy_fn):
        if not use_scores:
            return None
        if prob_path.score_type == "self":
            if model_energy_fn is None:
                raise ValueError("model_energy_fn is required when score_type='self'.")
            with torch.enable_grad():
                temp_x = x.detach().requires_grad_(True)
                return -torch.autograd.grad(
                    model_energy_fn(temp_x, t).sum(),
                    temp_x,
                    create_graph=False,
                )[0].detach()
        return prob_path.get_target_score(x=x, t=t).detach()

    def build_noise_fn(original_data, train=False, model_energy_fn=None):
        del train

        ts, taus = get_cont_times(
            batch_size=original_data.shape[0],
            time_std=time_std,
            prob_path=prob_path,
            device=original_data.device,
            rne=rne,
            resolve_collisions=resolve_collisions,
        )

        if estimator == "tnce":
            xs, ys = prob_path.sample2(original_data, ts, taus)

            return (
                torch.concat([xs, ys], dim=0),
                torch.concat([ts, taus], dim=0),
                torch.concat([taus, ts], dim=0),
            )

        elif estimator == "tcnce":
            xs, ys = prob_path.sample2(original_data, ts, taus)

            perturb_noises = noise_std * torch.randn_like(xs)
            xs_perturbed = xs + perturb_noises
            ys_perturbed = ys + perturb_noises
            all_times = torch.concat([ts, taus], dim=0)

            return (
                torch.concat([xs, ys], dim=0),
                all_times,
                torch.concat([xs_perturbed, ys_perturbed], dim=0),
                all_times,
                torch.zeros_like(all_times),
            )

        if estimator == "tstnce":
            taus, ts = torch.min(taus, ts), torch.max(taus, ts)

            if prob_path.stnce_noise_type in ["scale", "diff", "recovery-diff"]:
                # noise kernel coincides with forward process
                xs = prob_path.sample(original_data, ts)
                ys = prob_path.conditional_sample(xs, ts, taus)
                ys_scores = get_scores(ys, taus, model_energy_fn)

                rev_xs = prob_path.conditional_sample(ys, taus, ts, ys_scores)
                xs_ts_conds = prob_path.conditional_log_prob(
                    xs, ts, ys, taus
                ) - prob_path.conditional_log_prob(ys, taus, xs, ts, ys_scores)
                ys_taus_conds = prob_path.conditional_log_prob(
                    ys, taus, rev_xs, ts, ys_scores
                ) - prob_path.conditional_log_prob(rev_xs, ts, ys, taus)

                return (
                    xs,
                    ts,
                    ys,
                    taus,
                    rev_xs,
                    torch.concat([xs_ts_conds, ys_taus_conds], dim=0).unsqueeze(dim=-1),
                )

            elif prob_path.stnce_noise_type == "white":
                xs, ys = prob_path.sample2(original_data, ts, taus)
                perturb_noises = noise_std * torch.randn_like(xs)
                fwd_ys = xs + perturb_noises
                rev_xs = ys + perturb_noises
                all_conds = torch.zeros(
                    (2 * ts.shape[0], 1), dtype=ts.dtype, device=ts.device
                )

                return (
                    torch.concat([xs, ys], dim=0),
                    torch.concat([ts, taus], dim=0),
                    torch.concat([fwd_ys, rev_xs], dim=0),
                    torch.concat([taus, ts], dim=0),
                    all_conds,
                )

            else:
                xs, ys = prob_path.sample2(original_data, ts, taus)
                fwd_ys = prob_path.conditional_sample(xs, ts, taus)

                rev_xs = prob_path.conditional_sample(ys, taus, ts)
                xs_ts_conds = prob_path.conditional_log_prob(
                    xs, ts, fwd_ys, taus
                ) - prob_path.conditional_log_prob(fwd_ys, taus, xs, ts)
                ys_taus_conds = prob_path.conditional_log_prob(
                    ys, taus, rev_xs, ts
                ) - prob_path.conditional_log_prob(rev_xs, ts, ys, taus)

                return (
                    torch.concat([xs, ys], dim=0),
                    torch.concat([ts, taus], dim=0),
                    torch.concat([fwd_ys, rev_xs], dim=0),
                    torch.concat([taus, ts], dim=0),
                    torch.concat([xs_ts_conds, ys_taus_conds], dim=0).unsqueeze(dim=-1),
                )

        raise NotImplementedError

    return build_noise_fn


def get_combine_noise(
    config,
    prob_path,
    main_estimator,
    reg_estimator,
    main_build_noise_fn,
    reg_build_noise_fn,
):
    rne = config.estim.rne
    reg_main_prop = config.estim.reg_main_prop
    mixture_loss = getattr(config.estim, "mixture_loss", False)

    if mixture_loss:
        if (main_estimator != "tnce") or (reg_estimator != "tcnce"):
            raise ValueError(
                "mixture_loss=True is only supported for "
                "config.estim.name='tnce' and config.estim.reg_estimator='tcnce'."
            )
        if config.estim.combine_type not in ["default", "reuse"]:
            raise ValueError(
                "mixture_loss=True requires config.estim.combine_type to be "
                "'default' or 'reuse'."
            )

        if config.estim.combine_type == "reuse":
            mixture_main_build_noise_fn = get_reuse_main_noise(
                config=config,
                prob_path=prob_path,
                estimator=main_estimator,
            )
            mixture_reg_build_noise_fn = get_reuse_main_noise(
                config=config,
                prob_path=prob_path,
                estimator=reg_estimator,
            )
        else:
            mixture_main_build_noise_fn = main_build_noise_fn
            mixture_reg_build_noise_fn = reg_build_noise_fn

        def build_noise_fn(original_data, train=False, model_energy_fn=None):
            num_total = original_data.shape[0]
            if num_total == 0:
                raise ValueError("mixture_loss received an empty batch.")

            main_mask = (
                torch.rand(num_total, device=original_data.device) < reg_main_prop
            )
            num_main = int(main_mask.sum().item())

            main_noises = None
            if num_main > 0:
                main_noises = mixture_main_build_noise_fn(
                    original_data=original_data[main_mask],
                    train=train,
                    model_energy_fn=model_energy_fn,
                )

            reg_noises = None
            if num_main < num_total:
                reg_noises = mixture_reg_build_noise_fn(
                    original_data=original_data[~main_mask],
                    train=train,
                    model_energy_fn=model_energy_fn,
                )

            return (main_noises, reg_noises, num_main / num_total)

        return build_noise_fn

    if config.estim.combine_type == "separate":

        def build_noise_fn(original_data, train=False, model_energy_fn=None):
            num_total = original_data.shape[0]
            num_main = round(reg_main_prop * num_total)
            return (
                main_build_noise_fn(
                    original_data=original_data[:num_main],
                    train=train,
                    model_energy_fn=model_energy_fn,
                ),
                reg_build_noise_fn(
                    original_data=original_data[num_main:],
                    train=train,
                    model_energy_fn=model_energy_fn,
                ),
                num_main / num_total,
            )

    elif config.estim.combine_type == "reuse":
        # only works for continuous time

        time_std = config.estim.time_std

        resolve_collisions = config.estim.resolve_collisions

        @torch.enable_grad()
        def build_noise_fn(original_data, train=False, model_energy_fn=None):
            # TODO
            num_total = original_data.shape[0]
            # Here we sample xs, ts, ys, taus for DSM and evaluate the scores
            ts, taus = get_cont_times(
                batch_size=original_data.shape[0],
                time_std=time_std,
                prob_path=prob_path,
                device=original_data.device,
                rne=rne,
                resolve_collisions=resolve_collisions,
            )

            if main_estimator == "tnce":
                xs, ys, noises = prob_path.sample2_and_noises(original_data, ts, taus)

                noises = torch.concat([noises, noises], dim=0)
            elif main_estimator == "tstnce":
                # make taus <= ts
                taus, ts = torch.min(taus, ts), torch.max(taus, ts)

                # sample the larger time first then use forward process
                xs, xs_noises = prob_path.sample_and_noises(original_data, ts)
                ys = prob_path.conditional_sample(xs, ts, taus)
                ys_noises = prob_path.recover_noises(ys, original_data, taus)

                noises = torch.concat([xs_noises, ys_noises], dim=0)

            dsm_xs = torch.concat([xs, ys], dim=0)
            dsm_xs.requires_grad_(True)
            dsm_ts = torch.concat([ts, taus], dim=0)

            if main_estimator == "tnce":
                all_xs = torch.concat([dsm_xs, xs, ys], dim=0)
                all_ts = torch.concat([dsm_ts, taus, ts], dim=0)
                all_energies = model_energy_fn(all_xs, all_ts)
                num_dsm_total = dsm_xs.shape[0]
                dsm_scores = torch.autograd.grad(
                    all_energies[:num_dsm_total].sum(), dsm_xs, create_graph=train
                )[0]
                if not train:
                    dsm_xs.requires_grad_(False)
                    dsm_scores = dsm_scores.detach()
                    all_energies = all_energies.detach()
                # for main noise
                # (xs, ts) vs (xs, taus)
                main_energy_xs_ts = all_energies[:num_total]
                main_energy_xs_taus = all_energies[
                    num_dsm_total : num_dsm_total + num_total
                ]
                # (ys, taus) vs (ys, ts)
                main_energy_ys_taus = all_energies[num_total:num_dsm_total]
                main_energy_ys_ts = all_energies[num_dsm_total + num_total :]

                main_noises = (
                    torch.concat([main_energy_xs_ts, main_energy_ys_taus], dim=0),
                    torch.concat([main_energy_xs_taus, main_energy_ys_ts], dim=0),
                )

            elif main_estimator == "tstnce":
                dsm_energies = model_energy_fn(dsm_xs, dsm_ts)
                num_dsm_total = dsm_xs.shape[0]
                dsm_scores = torch.autograd.grad(
                    dsm_energies.sum(), dsm_xs, create_graph=train
                )[0]
                if not train:
                    dsm_xs.requires_grad_(False)
                    dsm_scores = dsm_scores.detach()
                    dsm_energies = dsm_energies.detach()
                # for main noise
                # (xs, ts) vs (ys, taus)
                main_energy_xs_ts = dsm_energies[:num_total]
                main_energy_ys_taus = dsm_energies[num_total:]

                if prob_path.score_type == "self":
                    ys_scores = (-dsm_scores[num_total:]).detach()
                elif prob_path.score_type in ["model", "oracle"]:
                    ys_scores = prob_path.get_target_score(ys, taus).detach()
                else:
                    ys_scores = None

                rev_xs = prob_path.conditional_sample(ys, taus, ts, ys_scores)

                with torch.set_grad_enabled(train):
                    main_energy_rev_xs_ts = model_energy_fn(rev_xs, ts)

                xs_ts_conds1 = prob_path.conditional_log_prob(xs, ts, ys, taus)
                ys_taus_conds2 = prob_path.conditional_log_prob(rev_xs, ts, ys, taus)

                xs_ts_conds2 = prob_path.conditional_log_prob(
                    ys, taus, xs, ts, ys_scores
                )
                ys_taus_conds1 = prob_path.conditional_log_prob(
                    ys, taus, rev_xs, ts, ys_scores
                )

                xs_ts_conds = (xs_ts_conds1 - xs_ts_conds2).unsqueeze(dim=-1)
                ys_taus_conds = (ys_taus_conds1 - ys_taus_conds2).unsqueeze(dim=-1)

                main_noises = (
                    torch.concat([main_energy_xs_ts, main_energy_ys_taus], dim=0),
                    torch.concat([main_energy_ys_taus, main_energy_rev_xs_ts], dim=0),
                    torch.concat([xs_ts_conds, ys_taus_conds], dim=0),
                )

            if reg_estimator == "dsm":
                with torch.no_grad():
                    _, dsm_targets, dsm_weights = prob_path.sample_and_stein_score(
                        z=None, t=dsm_ts, temp=[dsm_xs, noises]
                    )
                reg_noises = [dsm_targets, dsm_weights, dsm_scores]

            elif reg_estimator == "fm":
                with torch.no_grad():
                    _, fm_targets = prob_path.sample_and_velocities(
                        z=None, t=dsm_ts, temp=[dsm_xs, noises]
                    )
                reg_noises = [
                    fm_targets,
                    prob_path.v_from_score(-dsm_scores, dsm_xs, dsm_ts),
                ]

            return (main_noises, reg_noises, 0.5)

    elif config.estim.combine_type == "default":
        # only works for continuous time

        if (main_estimator == "ctsm") and (reg_estimator in ["dsm", "fm"]):
            # dual SM
            @torch.enable_grad()
            def build_noise_fn(original_data, train=False, model_energy_fn=None):
                ts = prob_path.sample_t(
                    original_data.shape[0], device=original_data.device
                )
                xs, noises = prob_path.sample_and_noises(original_data, ts)
                xs.requires_grad_(True)
                ts.requires_grad_(True)
                energies = model_energy_fn(xs, ts)
                dsm_scores, ctsm_scores = torch.autograd.grad(
                    outputs=energies.sum(), inputs=[xs, ts], create_graph=train
                )
                _, ctsm_targets, ctsm_weights = prob_path.sample_and_time_score(
                    original_data, ts, [xs, noises]
                )
                if reg_estimator == "dsm":
                    _, dsm_targets, dsm_weights = prob_path.sample_and_stein_score(
                        original_data, ts, [xs, noises]
                    )
                    reg_noise = [dsm_targets, dsm_weights, dsm_scores]
                elif reg_estimator == "fm":
                    _, fm_targets = prob_path.sample_and_velocities(
                        original_data, ts, [xs, noises]
                    )
                    reg_noise = [
                        fm_targets,
                        prob_path.v_from_score(-dsm_scores, xs, ts),
                    ]
                return (
                    [ctsm_targets, ctsm_weights, ctsm_scores],
                    reg_noise,
                    0.5,
                )

        else:

            def build_noise_fn(original_data, train=False, model_energy_fn=None):
                ts = prob_path.sample_t(
                    original_data.shape[0], device=original_data.device
                )
                xs, noises = prob_path.sample_and_noises(original_data, ts)
                return (
                    main_build_noise_fn(
                        original_data=original_data,
                        train=train,
                        temp=[xs, ts, noises],
                        model_energy_fn=model_energy_fn,
                    ),
                    reg_build_noise_fn(
                        original_data=original_data,
                        train=train,
                        temp=[xs, ts, noises],
                        model_energy_fn=model_energy_fn,
                    ),
                    xs,
                    ts,
                    0.5,
                )

    return build_noise_fn
